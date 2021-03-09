# C2C-GenDA

## Introduction

We propose a novel Cluster-to-Cluster generation framework for Data Augmentation (DA) of slot filling, named C2C-GenDA.

For a detailed description and experimental results, please refer to our paper: [C2C-GenDA: Cluster-to-Cluster Generation for Data Augmentation of Slot Filling](https://arxiv.org/abs/2012.07004) (Accepted by AAAI-2021).

## Environment
Python 3.6, Pytorch 0.4.1, Pytorch-nlp 0.4.1

## Get Started
1. Construct cluster-to-cluster 'paraphrasing' pairs from original training data.
    ```bash
    python run_data_preparation.py \
          --data_path ./data/Atis/atis_train \
          --output_cluster_dir ./data/Cluster \
          --output_pairs_dir ./data/AugmentedData \
          --num_per_source_cluster 5 \
          --target_cluster_num_threshold 10 \
          --cross_generation \
          --debug_mode 
    ```
    **Note**: We set the '--num_per_cluster' (avg. num per source cluster) to 5/3/2 for full/medium/small proportion respectively.
    
    After construction, you can find (1) *'the classified data file'*, *'the classified and clustered data file'*, *'the slot2entity dictionary file'* and *'the file including all the sentences'* at './data/Cluster/'.
    (2) the cluster-to-cluster pairs for training and generation at './data/AugmentedData'.
    
2. Fine-tune GPT2 model with the cluster-to-cluster 'paraphrasing' pairs constructed in the first step.
    ```bash
    python run_C2C-GenDA_with_gpt2.py \
          --model_name /PATH/TO/PRETRAINED/GPT2 \
          --do_train \
          --log_output_path ./log.txt \
          --model_output_dir ./model \
          --train_dataset ./data/AugmentedData/atis_train_clustered_train0.txt \
          --slots_tokens_path ./data/Cluster/atis_train_slot2entity_dictionary.json \
          --target_cluster_num_threshold 10 \
          --unknown_token '<UNK>' \
          --num_train_epochs 10 \
          --train_batch_size 1 \
          --train_target_size 5 \
          --intra_kl_loss \
          --intra_kl_loss_weight 1 \
          --intra_kl_anneal_func constant \
          --intra_attention \
          --intra_attention_weight 0.01 
    ```
    **Note**: if the path of the cluster-to-cluster source and target pairs for training are './train_src.txt' and 'tran_tgt.txt' respectively, then set the '--train_dataset' to the path './train.txt'.
    
    After fine-tuning, you can find your log file at './log.txt', and your model checkpoints at './model/'.

3. Generate new data with the Cluster2Cluster generation model trained in the second step.
    ```bash
    python run_C2C-GenDA_with_gpt2.py \
          --model_name ./model/checkpoint/ \
          --do_gen \
          --log_output_path ./log.txt \
          --gen_output_dir ./data/AugmentedData/gen/ \
          --gen_dataset ./data/AugmentedData/atis_train_clustered_reserve0.txt \
          --slots_tokens_path ./data/Cluster/atis_train_slot2entity_dictionary.json \
          --original_data_path ./data/Cluster/atis_train_all_sentences.txt \
          --intra_attention \
          --intra_attention_weight 0.01 \
          --gen_length 40 \
          --gen_mode sample \
          --gen_accept_empty \
          --gen_stop_early 
    ```
    **Note**: if the path of the reserved source file is './reserve_src.txt', then set the '--gen_dataset' to the path './reserve.txt'. After generating, you'll find a generated file named 'reserve_gen.txt' at './data/AugmentedData/gen/'.
    
    After generating, you can find your log file at './log.txt'.

4. Surface Realization: replace each special slot token with context-suitable values.
    ```bash
    python SurfaceRealization/run_surface_realization.py \
            --input_path data/AugmentedData/gen/atis_train_augmented \
            --values_path data/Cluster/atis_train_slot2entity_dictionary.json \
            --output_path data/AugmentedData/gen/atis_train_augmented_surface_realized.json
    ```
    
    After surface-realization, you can find your augmented dataset at 'data/AugmentedData/gen/atis_train_augmented_surface_realized.json'.

5. Slot Filling: we evaluate our C2C-GenDA with the Bi-LSTM slot filling model implemented [here](https://github.com/AtmaHou/Bi-LSTM_PosTagger), and train it with both the original training data and the augmented data generated in the forth step.

## Citation
If you find our C2C-GenDA useful, please cite [our paper](https://arxiv.org/abs/2012.07004):
```bibtex
@article{C2C-GenDA,
  title={C2C-GenDA: Cluster-to-Cluster Generation for Data Augmentation of Slot Filling},
  author={Hou, Yutai and Chen, Sanyuan and Che, Wanxiang and Chen, Cheng and Liu, Ting},
  journal={arXiv preprint arXiv:2012.07004},
  year={2020}
}
```