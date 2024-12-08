# FGTI-MSA
Code for FGTI:A FINE-GRAINED TRI-MODAL INTERACTION MODEL FOR MULTIMODAL SENTIMENT ANALYSIS(ICASSP 2024).

This project is built upon https://github.com/thuiar/MMSA .

The Hilbert-schmidt independence criteria(HSIC) is based on [Hilbert-schmidt independence criteria](https://github.com/xiao-he/HSIC) .

# Run
- CMU-MOSI
`python src/train_run.py`
- CMU-MOSEI
`python src/train_run_mosei.py`
# Overview


The main contributions can be summarised as follows: 

(1) We propose a novel fine-grained multimodal sentiment analysis model that works with complex multimodal data through dual-stream encoding. The supervised contrastive learning and label constraints make the modal focus more dynamics inter/intra label-level forms.

(2) Through multigranularity modal interaction, the ensemble from multimodalities is adopted to correct fine-grained deviations and perform complete cross-modal fusion. 

(3) We conduct a series of experiments on two datasets, showing that the performance of the proposed model exceeds that of state-of-the-art methods.

# Citation
If you think this project is helpful, please feel free to leave a star and cite our paper:
```
@INPROCEEDINGS{10447872,
  author={Zhi, Yuxing and Li, Junhuai and Wang, Huaijun and Chen, Jing and Cao, Ting},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={A Fine-Grained Tri-Modal Interaction Model for Multimodal Sentiment Analysis}, 
  year={2024},
  pages={5715-5719},
  doi={10.1109/ICASSP48485.2024.10447872}
}
```
