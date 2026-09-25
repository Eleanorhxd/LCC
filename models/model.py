import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from transformers import LlamaForCausalLM, LlamaTokenizer, DynamicCache
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor
from transformers import SwinModel
from lightning_tools.optim import config_optimizer
from peft import get_peft_model, LoraConfig, TaskType
from mertic.mertic import compute_all_scores
from typing import Optional, Dict, Any, List, Tuple



DEFAULT_ARGS = {
    'chexbert_path': './checkpoints/pretrained/chexbert/chexbert.pth',
    'bert_path': './checkpoints/bert-base-uncased',
    'radgraph_path': './checkpoints/radgraph.tar.gz',
}
class LCC(pl.LightningModule):
    """
    LCC model.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.compute_all_scores = compute_all_scores

        print(f'Loading vision encoder:{args.vision_model}')
        self.visual_encoder = SwinModel.from_pretrained(args.vision_model)
        if args.vis_use_lora:
            peft_config_visual = LoraConfig(
                                    r=args.vis_r,
                                    lora_alpha=args.vis_alpha,
                                    target_modules=["query", "value"],
                                    lora_dropout=args.lora_dropout,
                                    bias="none",
                                    modules_to_save=["classifier"],
                                )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
            self.visual_encoder.print_trainable_parameters()
            print('Loading vision encoder with LoRA -- Done')
        elif args.freeze_vm:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            print(f'Loading Frozen vision encoder:{args.vision_model} -- Done')
        else:
            print(f'Loading Trainable vision encoder:{args.vision_model} -- Done')

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(args.llama_model, use_fast=False)
        self.llama_tokenizer.pad_token_id = 0
        if args.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map="auto"
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype=torch.float16,
            )
         
        if args.llm_use_lora:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.llm_r, lora_alpha=args.llm_alpha, lora_dropout=args.lora_dropout
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
            print('Loading LLAMA LoRA Done')         
        else:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            print('Loading LLAMA Done')

        self.llama_proj = nn.Linear(self.visual_encoder.num_features, self.llama_model.config.hidden_size)
        self.layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)
        self.end_sym = args.end_sym
        self.prompt = 'Generate a comprehensive and detailed diagnosis report for this chest xray image.'
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0
        
     
        self.use_latent_morph = getattr(args, 'use_latent_morph', False)
        self.latent_morph_controller = None
        
        if self.use_latent_morph:
            print('=' * 60)
            print('Initializing LatentMorph Integration (Enhanced)...')
            print('=' * 60)
            
            hidden_dim = self.llama_model.config.hidden_size
           
            vision_dim_for_controller = hidden_dim
            
           
            lm_config = LatentMorphReportConfig(
                enabled=True,
                d_model=hidden_dim,
                d_vision=vision_dim_for_controller,
                d_condenser=getattr(args, 'latent_morph_d_condenser', 512),
                d_latent=getattr(args, 'latent_morph_d_latent', 256),
                check_every=getattr(args, 'latent_morph_check_every', 16),
                max_triggers=getattr(args, 'latent_morph_max_triggers', 3),
                min_trigger_position=getattr(args, 'latent_morph_min_trigger_position', 32),
                num_control_tokens=getattr(args, 'latent_morph_num_control_tokens', 4),
                control_strength=getattr(args, 'latent_morph_control_strength', 1.0),
                injection_threshold=getattr(args, 'latent_morph_injection_threshold', 0.5),
                # 视觉潜在推理
                use_vision_latent_reasoning=getattr(args, 'latent_morph_use_vision_reasoning', True),
                num_vision_reasoning_steps=getattr(args, 'latent_morph_vision_reasoning_steps', 3),
                vision_reasoning_heads=getattr(args, 'latent_morph_vision_reasoning_heads', 8),
                # 跨模态对齐
                use_cross_modal_alignment=getattr(args, 'latent_morph_use_cross_modal', True),
                alignment_temperature=getattr(args, 'latent_morph_alignment_temp', 0.07),
                # 消融：双语义凝缩
                use_dual_condensation=getattr(args, 'latent_morph_use_dual_condensation', True),
            )
            
           
            self.latent_morph_controller = LatentMorphReportController(lm_config)
            
          
            train_controller_only = getattr(args, 'latent_morph_train_controller_only', False)

        # 多阶段在这里设置
        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))['model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')
    


    def score(self, ref, hypo):
        """
        ref, dictionary of reference sentences (id, sentence)
        hypo, dictionary of hypothesis sentences (id, sentence)
        score, dictionary of scores
        """
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Meteor(), "METEOR"),
            (Cider(), "CIDEr")
        ]
        final_scores = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores


    def encode_img(self, images):
        image_embeds = []
        for image in images:
            device = image.device
            if self.hparams.global_only:
                image_embed = self.visual_encoder(image)['pooler_output'].unsqueeze(1).to(device)
            else:
                image_embed = self.visual_encoder(image)['last_hidden_state'].to(device)
            image_embeds.append(image_embed)
            
        image_embeds = torch.stack(image_embeds).mean(0)
        inputs_llama = self.llama_proj(image_embeds)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama


    def prompt_wrap(self, img_embeds, atts_img):
        prompt=f'Human: <Img><ImageHere></Img> {self.prompt} \nAssistant:'
        batch_size = img_embeds.shape[0]
        p_before, p_after = prompt.split('<ImageHere>')
        p_before_tokens = self.llama_tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_after_tokens = self.llama_tokenizer(
            p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
        return wrapped_img_embeds, wrapped_atts_img


    def forward(self, samples):
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        
        device = image[0].device
        batch_size = img_embeds.shape[0]

        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["input_text"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        ).to(device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == 0, -100
        )
        
       
        if self.use_latent_morph and self.latent_morph_controller is not None:
            # 使用 LatentMorph 进行训练
            return self._forward_with_latent_morph(
                img_embeds, atts_img, to_regress_tokens, targets, device, batch_size
            )
        
       
        else:
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

            empty_targets = (
                torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                           dtype=torch.long).to(device).fill_(-100)  # plus one for bos
            )
            targets = torch.cat([empty_targets, targets], dim=1)

            bos = torch.ones([batch_size, 1],
                             dtype=to_regress_tokens.input_ids.dtype,
                             device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
            bos_embeds = self.embed_tokens(bos)
            atts_bos = atts_img[:, :1]

            to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
            inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
            attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss
            return {"loss": loss}

    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log_dict(result, prog_bar=True)
        return result

    def save_checkpoint(self, eval_res):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in param_grad_dic.keys():
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step":global_step
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'checkpoints',
            "checkpoint_epoch{}_step{}_bleu{:3f}_cider{:3f}.pth".format(current_epoch, global_step, eval_res['Bleu_4'], eval_res['CIDEr']),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)
    
    def validation_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        
      
        if self.use_latent_morph and self.latent_morph_controller is not None:
            outputs = self._generate_with_latent_morph(
                img_embeds, atts_img, 
                max_new_tokens=self.hparams.max_new_tokens,
            )
        else:
            # 标准生成流程
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)
            batch_size = img_embeds.shape[0]
            bos = torch.ones([batch_size, 1],
                             dtype=atts_img.dtype,
                             device=atts_img.device) * self.llama_tokenizer.bos_token_id
            bos_embeds = self.embed_tokens(bos)
            atts_bos = atts_img[:, :1]

            inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
            attention_mask = torch.cat([atts_bos, atts_img], dim=1)

            outputs = self.llama_model.generate(
                inputs_embeds=inputs_embeds,
                num_beams=self.hparams.beam_size,
                do_sample=self.hparams.do_sample,
                min_new_tokens=self.hparams.min_new_tokens,
                max_new_tokens=self.hparams.max_new_tokens,
                repetition_penalty=self.hparams.repetition_penalty,
                length_penalty=self.hparams.length_penalty,
                temperature=self.hparams.temperature,
            )
        
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        
        self.val_step_outputs.append({
            "hypo": hypo, 
            "ref": ref, 
            "id": samples["id"]
        })
        return hypo, ref
    
    def decode(self, output_token):
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('</s>')[0].strip()
        output_text = output_text.replace('<unk>', '')
        return output_text

    def on_validation_epoch_end(self):
        ref, hypo, ids = [], [], []
        
        for i in self.val_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}
        
        eval_res = self.compute_all_scores(gts=ref, res=hypo, args=DEFAULT_ARGS)
        
       
        self.log_dict(eval_res, sync_dist=True, logger=True)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'refs.json'), 'w'))
        
        self.print(eval_res)

        val_score = 0
        for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
            val_score += eval_res[score_type] * weight

        if self.trainer.local_rank == 0:
            if val_score > self.val_score:
                self.save_checkpoint(eval_res)
                self.val_score = val_score
        self.val_step_outputs.clear()


    def test_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)

      
        if self.use_latent_morph and self.latent_morph_controller is not None:
            outputs = self._generate_with_latent_morph(
                img_embeds, atts_img,
                max_new_tokens=self.hparams.max_new_tokens,
            )
        else:
            
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)
            batch_size = img_embeds.shape[0]
            bos = torch.ones([batch_size, 1],
                             dtype=atts_img.dtype,
                             device=atts_img.device) * self.llama_tokenizer.bos_token_id
            bos_embeds = self.embed_tokens(bos)
            atts_bos = atts_img[:, :1]

            inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
            attention_mask = torch.cat([atts_bos, atts_img], dim=1)

            outputs = self.llama_model.generate(
                inputs_embeds=inputs_embeds,
                num_beams=self.hparams.beam_size,
                do_sample=self.hparams.do_sample,
                min_new_tokens=self.hparams.min_new_tokens,
                max_new_tokens=self.hparams.max_new_tokens,
                repetition_penalty=self.hparams.repetition_penalty,
                length_penalty=self.hparams.length_penalty,
                temperature=self.hparams.temperature,
            )
        
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        
        self.test_step_outputs.append({
            "hypo": hypo, 
            "ref": ref, 
            "id": samples["id"]
        })
        return hypo, ref


    def on_test_epoch_end(self):
   
        ref, hypo, ids = [], [], []
        
        for i in self.test_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}
        
        
        eval_res = self.compute_all_scores(gts=ref, res=hypo, args=DEFAULT_ARGS)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        json.dump(hypo, open(os.path.join(result_folder, f"test_result.json"), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'test_refs.json'), 'w'))
        
        self.print(f"Test result of {self.hparams.delta_file}: {eval_res}")

 
    
    def _forward_with_latent_morph(
        self,
        img_embeds: torch.Tensor,
        atts_img: torch.Tensor,
        to_regress_tokens,
        targets: torch.Tensor,
        device: torch.device,
        batch_size: int,
    ) -> Dict[str, torch.Tensor]:

        controller = self.latent_morph_controller
        controller.reset()
        
        
        img_embeds_wrapped, atts_img_wrapped = self.prompt_wrap(img_embeds, atts_img)
        
        
        bos = torch.ones([batch_size, 1], dtype=torch.long, device=device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img_wrapped[:, :1]
        
        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds_wrapped, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img_wrapped, to_regress_tokens.attention_mask], dim=1)
        
       
        empty_targets = torch.ones(
            [atts_img_wrapped.shape[0], atts_img_wrapped.shape[1] + 1],
            dtype=torch.long, device=device
        ).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)
        
       
        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            labels=targets,
        )
        
        lm_loss = outputs.loss
        
       
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
        
        
        text_start_idx = 1 + img_embeds_wrapped.shape[1]
        text_hidden = hidden_states[:, text_start_idx:, :]  # [batch_size, text_len, hidden_dim]
        
    
        if text_hidden.shape[1] > controller.cfg.min_trigger_position:
           
            latent_morph_losses = controller.compute_training_losses(
                img_embeds=img_embeds,
                text_hidden=text_hidden,
            )
            
          
            control_result = controller.compute_control_signal(
                img_embeds=img_embeds,
                text_hidden=text_hidden,
            )
            
           
            consistency_loss = latent_morph_losses.get("consistency_loss", torch.tensor(0.0, device=device))
            alignment_loss = latent_morph_losses.get("alignment_loss", torch.tensor(0.0, device=device))
            control_reg_loss = latent_morph_losses.get("control_reg_loss", torch.tensor(0.0, device=device))
            
            latent_morph_loss = consistency_loss + alignment_loss + control_reg_loss
            
            # 总损失
            latent_morph_weight = getattr(self.hparams, 'latent_morph_loss_weight', 0.1)
            total_loss = lm_loss + latent_morph_weight * latent_morph_loss
            
            result = {
                "loss": total_loss,
                "lm_loss": lm_loss,
                "consistency_loss": consistency_loss,
                "consistency_score": control_result["consistency_score"].mean(),
            }
            
           
            if "alignment_score" in control_result:
                result["alignment_score"] = control_result["alignment_score"].mean()
            if "alignment_loss" in latent_morph_losses:
                result["alignment_loss"] = alignment_loss
            
            return result
        else:
            return {"loss": lm_loss}
    
    def _generate_with_latent_morph(
        self,
        img_embeds: torch.Tensor,
        atts_img: torch.Tensor,
        max_new_tokens: int = 120,
    ) -> torch.Tensor:
    
        controller = self.latent_morph_controller
        controller.reset()
        
        device = img_embeds.device
        batch_size = img_embeds.shape[0]
  
        img_embeds_wrapped, atts_img_wrapped = self.prompt_wrap(img_embeds, atts_img)

        bos = torch.ones([batch_size, 1], dtype=torch.long, device=device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        
        current_embeds = torch.cat([bos_embeds, img_embeds_wrapped], dim=1)
        
      
        generated_ids = []
        past_key_values = None
        
        hidden_states_buffer = []

        num_control_injections = 0
        avg_consistency = []
        avg_alignment = []
        
        for step in range(max_new_tokens):
           
            if past_key_values is None:
                outputs = self.llama_model(
                    inputs_embeds=current_embeds,
                    output_hidden_states=True,
                    use_cache=True,
                )
            else:
                outputs = self.llama_model(
                    inputs_embeds=current_embeds,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    use_cache=True,
                )
            
            past_key_values = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1:, :]  # [batch_size, 1, hidden_dim]
            hidden_states_buffer.append(last_hidden)
            
            # 获取 logits 并采样
            logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)  # [batch_size, 1]
            generated_ids.append(next_token)
            
            
            use_cc = getattr(self.hparams, 'latent_morph_use_consistency_check_decode', True)
            allow_inject = getattr(self.hparams, 'latent_morph_allow_control_injection_decode', True)
            if use_cc and controller.should_trigger(step):

                text_hidden = torch.cat(hidden_states_buffer, dim=1)  # [batch_size, step+1, hidden_dim]

                control_result = controller.compute_control_signal(
                    img_embeds=img_embeds,
                    text_hidden=text_hidden,
                )
                
                control_tokens = control_result["control_tokens"]
                consistency_score = control_result["consistency_score"]
                should_inject = control_result["should_inject"] if allow_inject else torch.zeros_like(control_result["should_inject"], dtype=torch.bool, device=control_result["should_inject"].device)
                alignment_score = control_result.get("alignment_score", consistency_score)
                
        
                avg_consistency.append(consistency_score.mean().item())
                avg_alignment.append(alignment_score.mean().item())
   
            current_embeds = self.embed_tokens(next_token)
        
        # 合并生成的 token IDs
        if generated_ids:
            output_ids = torch.cat(generated_ids, dim=1)
        else:
            output_ids = torch.zeros([batch_size, 1], dtype=torch.long, device=device)
        
        return output_ids

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()
    