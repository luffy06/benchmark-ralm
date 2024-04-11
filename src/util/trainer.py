import logging
import transformers

logger = logging.getLogger(__name__)

class Trainer(transformers.Trainer):
    # Inherit from transformers.Trainer (version 4.40.0)
    def bilevel_train(self):
        raise NotImplementedError("Bilevel training is not implemented yet")
        tr_loss = 0
        bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
        eval_iterator = iter(eval_dataloader)
        for step, batch in enumerate(bar):
            for eval_step in range(self.args.gradient_accumulation_steps):
                try:
                    eval_batch = next(eval_iterator)
                except StopIteration:
                    eval_iterator = iter(eval_dataloader)
                    eval_batch = next(eval_iterator)
                eval_batch = {k: v.to(self.args.device) if hasattr(v, "to") else v for k, v in eval_batch.items()}
                eval_source_ids, eval_target_ids = eval_batch["input_ids"], eval_batch["labels"]
                eval_source_mask = eval_batch["attention_mask"]
                eval_target_mask = eval_batch["decoder_attention_mask"] if "decoder_attention_mask" in eval_batch else None
                
                if model.model_args.transformer_type == "encoder-decoder":
                    eval_outputs = model(
                        input_ids=eval_source_ids, 
                        attention_mask=eval_source_mask,
                        labels=eval_target_ids, 
                        decoder_attention_mask=eval_target_mask,
                        neighbors=eval_batch["neighbors"],
                        neighbor_texts=eval_batch["neighbor_texts"],
                    )
                else:
                    eval_outputs = model(
                        input_ids=eval_source_ids,
                        attention_mask=eval_source_mask,
                        labels=eval_target_ids, 
                        neighbors=eval_batch["neighbors"],
                        neighbor_texts=eval_batch["neighbor_texts"],
                    )

                eval_loss = eval_outputs.loss
                if self.args.n_gpu > 1:
                    eval_loss = eval_loss.mean()
                if self.args.gradient_accumulation_steps > 1:
                    eval_loss = eval_loss / self.args.gradient_accumulation_steps
                eval_loss.backward()
            arch_optimizer.step()
            arch_scheduler.step()
            arch_optimizer.zero_grad()

            batch = {k: v.to(self.args.device) if hasattr(v, "to") else v for k, v in batch.items()}
            source_ids, target_ids = batch["input_ids"], batch["labels"]
            source_mask = batch["attention_mask"]
            target_mask = batch["decoder_attention_mask"] if "decoder_attention_mask" in batch else None

            if model.model_args.transformer_type == "encoder-decoder":
                outputs = model(
                    input_ids=source_ids, 
                    attention_mask=source_mask,
                    labels=target_ids, 
                    decoder_attention_mask=target_mask,
                    neighbors=batch["neighbors"],
                    neighbor_texts=batch["neighbor_texts"]
                )
            else:
                outputs = model(
                    input_ids=source_ids, 
                    attention_mask=source_mask,
                    labels=target_ids, 
                    neighbors=batch["neighbors"],
                    neighbor_texts=batch["neighbor_texts"]
                )
            loss = outputs.loss
            
            if self.args.n_gpu > 1:
                loss = loss.mean()
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            tr_loss += loss.item()
            loss.backward()

            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                # Update parameters
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()