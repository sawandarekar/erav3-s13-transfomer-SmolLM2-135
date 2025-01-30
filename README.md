# erav3-s13-transfomer-SmolLM2-135
Reverse engineer SmolLM2-135 model

## Model Architecture
- Vocabulary Size: 49152
- Hidden Size: 576
- Number of Attention Heads: 9
- Intermediate Size: 1536
- Number of Hidden Layers: 30
- Activation Function: SiLU

## File Structure
. 
├── input.txt 
├── model.py 
├── README.md 
├── SmolLM2-135.yaml 
└── train.py

## Training
To train the model, run the following command:
```sh
python train.py
```

## Logs for 5000
```
(venv) sawan.darekar@MAC-XHXW23XJ7N erav3-s13-transfomer-SmolLM2-135 % python train_n.py
2025-01-29 15:33:51,647 - INFO - Using device: cpu
2025-01-29 15:33:53,984 - INFO - 
Model Architecture:
2025-01-29 15:33:53,984 - INFO - ==================================================
2025-01-29 15:33:53,984 - INFO - LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(49152, 576)
    (layers): ModuleList(
      (0-29): 30 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(in_features=576, out_features=576, bias=False)
          (k_proj): Linear(in_features=576, out_features=192, bias=False)
          (v_proj): Linear(in_features=576, out_features=192, bias=False)
          (o_proj): Linear(in_features=576, out_features=576, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=576, out_features=1536, bias=False)
          (up_proj): Linear(in_features=576, out_features=1536, bias=False)
          (down_proj): Linear(in_features=1536, out_features=576, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=576, out_features=49152, bias=False)
)
2025-01-29 15:33:53,986 - INFO - ==================================================
2025-01-29 15:33:53,986 - INFO - 
Total trainable parameters: 134,515,008 (134.52M)
2025-01-29 15:33:53,986 - INFO - ==================================================

2025-01-29 15:33:54,432 - INFO - Set pad_token to eos_token
2025-01-29 15:33:54,455 - INFO - Loaded 7222 text segments from input.txt
2025-01-29 15:33:54,456 - INFO - No checkpoint found at 5000 steps. Starting fresh training.
Training (loss: 12.0625):  10%|█████████████████▏                                                                                                                                                          | 500/5000 [1:27:26<11:58:28,  9.58s/it]2025-01-29 17:01:21,508 - INFO - 
Sample generation at step 500:
2025-01-29 17:01:21,512 - INFO - Prompt: Once upon a time, in a distant galaxy
2025-01-29 17:01:21,512 - INFO - Generated: Once upon a time, in a distant galaxy such<|endoftext|>

2025-01-29 17:01:23,666 - INFO - Checkpoint saved at step 500
Training (loss: 7.7959):  20%|██████████████████████████████████▌                                                                                                                                          | 1000/5000 [2:43:16<9:50:25,  8.86s/it]2025-01-29 18:17:22,085 - INFO - 
Sample generation at step 1000:
2025-01-29 18:17:22,087 - INFO - Prompt: Once upon a time, in a distant galaxy
2025-01-29 18:17:22,087 - INFO - Generated: Once upon a time, in a distant galaxyfeit for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for for forhostle should should shouldile thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou thou

2025-01-29 18:17:24,174 - INFO - Checkpoint saved at step 1000
Training (loss: 7.2111):  30%|███████████████████████████████████████████████████▉                                                                                                                         | 1500/5000 [4:00:07<9:09:45,  9.42s/it]2025-01-29 19:34:02,778 - INFO - 
Sample generation at step 1500:
2025-01-29 19:34:02,779 - INFO - Prompt: Once upon a time, in a distant galaxy
2025-01-29 19:34:02,779 - INFO - Generated: Once upon a time, in a distant galaxyre heard heard heard heard heard heard for death;;;;;;;;<|endoftext|>

2025-01-29 19:34:04,716 - INFO - Checkpoint saved at step 1500
Training (loss: 6.5209):  40%|█████████████████████████████████████████████████████████████████████▏                                                                                                       | 2000/5000 [5:12:47<7:28:30,  8.97s/it]2025-01-29 20:46:42,668 - INFO - 
Sample generation at step 2000:
2025-01-29 20:46:42,670 - INFO - Prompt: Once upon a time, in a distant galaxy
2025-01-29 20:46:42,670 - INFO - Generated: Once upon a time, in a distant galaxyanus hope sovereign you<|endoftext|>

2025-01-29 20:46:44,536 - INFO - Checkpoint saved at step 2000
Training (loss: 6.1976):  47%|████████████████████████████████████████████████████████████████████████████████▉                                                                                            | 2338/5000 [5:59:25<5:08:14,  6.95s/it]Training (loss: 6.1976):  47%|████████████████████████████████████████████████████████████████████████████████▉                                                                                            | 2339/5000 [5:59:32<5:10:49,  7.01s/it]Training (loss: 6.1976):  47%|████████████████████████████████████████████████████████████████████████████████▉                                                                                            | 2341/5000 [5:59:49<5:39:50,  7.67s/it]Training (loss: 8.8682):  50%|██████████████████████████████████████████████████████████████████████████████████████▌                                                                                      | 2500/5000 [6:22:42<5:56:13,  8.55s/it]2025-01-29 21:56:37,411 - INFO - 
Sample generation at step 2500:
2025-01-29 21:56:37,413 - INFO - Prompt: Once upon a time, in a distant galaxy
2025-01-29 21:56:37,413 - INFO - Generated: Once upon a time, in a distant galaxy never never never never<|endoftext|>

2025-01-29 21:56:39,381 - INFO - Checkpoint saved at step 2500
Training (loss: 5.6602):  60%|███████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                     | 3000/5000 [7:37:01<5:10:56,  9.33s/it]2025-01-29 23:11:03,462 - INFO - 
Sample generation at step 3000:
2025-01-29 23:11:03,464 - INFO - Prompt: Once upon a time, in a distant galaxy
2025-01-29 23:11:03,464 - INFO - Generated: Once upon a time, in a distant galaxy YUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUTUCUCUTUTUTUTUTUTUT

2025-01-29 23:11:05,436 - INFO - Checkpoint saved at step 3000
Training (loss: 5.2730):  70%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                    | 3500/5000 [8:52:22<3:54:13,  9.37s/it]2025-01-30 00:26:16,622 - INFO - 
Sample generation at step 3500:
2025-01-30 00:26:16,623 - INFO - Prompt: Once upon a time, in a distant galaxy
2025-01-30 00:26:16,623 - INFO - Generated: Once upon a time, in a distant galaxy<|endoftext|>

2025-01-30 00:26:18,586 - INFO - Checkpoint saved at step 3500
Training (loss: 5.3446):  80%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                  | 4000/5000 [10:02:33<2:22:57,  8.58s/it]2025-01-30 01:36:30,577 - INFO - 
Sample generation at step 4000:
2025-01-30 01:36:30,579 - INFO - Prompt: Once upon a time, in a distant galaxy
2025-01-30 01:36:30,579 - INFO - Generated: Once upon a time, in a distant galaxyiediedLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL not not not not not<|endoftext|>

2025-01-30 01:36:32,495 - INFO - Checkpoint saved at step 4000
Training (loss: 4.9296):  87%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                      | 4344/5000 [15:27:07<1:22:33,  7.55s/it]Training (loss: 4.9296):  87%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                      | 4347/5000 [15:27:30<1:21:22,  7.48s/it]Training (loss: 5.2801):  90%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                 | 4500/5000 [15:47:38<1:07:56,  8.15s/it]2025-01-30 07:21:32,889 - INFO - 
Sample generation at step 4500:
2025-01-30 07:21:32,891 - INFO - Prompt: Once upon a time, in a distant galaxy
2025-01-30 07:21:32,891 - INFO - Generated: Once upon a time, in a distant galaxyGood had say or<|endoftext|>

2025-01-30 07:21:34,770 - INFO - Checkpoint saved at step 4500
Training (loss: 5.4114): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [17:01:03<00:00,  9.11s/it]2025-01-30 08:34:57,637 - INFO - 
Sample generation at step 5000:
2025-01-30 08:34:57,639 - INFO - Prompt: Once upon a time, in a distant galaxy
2025-01-30 08:34:57,639 - INFO - Generated: Once upon a time, in a distant galaxy<|endoftext|>

2025-01-30 08:34:59,555 - INFO - Checkpoint saved at step 5000
Training (loss: 5.4114): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [17:01:05<00:00, 12.25s/it]
2025-01-30 08:35:00,965 - INFO - Checkpoint saved at step 5000
2025-01-30 08:35:00,965 - INFO - Training completed!
(venv) sawan.darekar@MAC-XHXW23XJ7N erav3-s13-transfomer-SmolLM2-135 %   
```

## Logs for re-run 50 epochs
```
(venv) sawan.darekar@MAC-XHXW23XJ7N erav3-s13-transfomer-SmolLM2-135 % python train_n.py
2025-01-30 08:37:30,402 - INFO - Using device: cpu
2025-01-30 08:37:32,737 - INFO - 
Model Architecture:
2025-01-30 08:37:32,737 - INFO - ==================================================
2025-01-30 08:37:32,737 - INFO - LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(49152, 576)
    (layers): ModuleList(
      (0-29): 30 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(in_features=576, out_features=576, bias=False)
          (k_proj): Linear(in_features=576, out_features=192, bias=False)
          (v_proj): Linear(in_features=576, out_features=192, bias=False)
          (o_proj): Linear(in_features=576, out_features=576, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=576, out_features=1536, bias=False)
          (up_proj): Linear(in_features=576, out_features=1536, bias=False)
          (down_proj): Linear(in_features=1536, out_features=576, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=576, out_features=49152, bias=False)
)
2025-01-30 08:37:32,738 - INFO - ==================================================
2025-01-30 08:37:32,738 - INFO - 
Total trainable parameters: 134,515,008 (134.52M)
2025-01-30 08:37:32,739 - INFO - ==================================================

2025-01-30 08:37:36,054 - INFO - Set pad_token to eos_token
2025-01-30 08:37:36,077 - INFO - Loaded 7222 text segments from input.txt
2025-01-30 08:37:36,077 - INFO - Found checkpoint at 5000 steps. Loading and training for 50 more steps.
Training: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5050/5050 [07:06<00:00,  8.53s/it]
2025-01-30 08:44:46,443 - INFO - Checkpoint saved at step 5050
2025-01-30 08:44:46,444 - INFO - Training completed!
(venv) sawan.darekar@MAC-XHXW23XJ7N erav3-s13-transfomer-SmolLM2-135 % 

```