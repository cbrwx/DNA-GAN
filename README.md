# DNA-GAN
DNA GAN is an advanced Generative Adversarial Network (GAN) tailored to generate DNA sequences based on environmental factors such as temperature, pH, chemical exposure, and radiation. It can be used for various purposes, such as generating DNA sequences that can survive in specific environments or for research purposes.

# Features
- Environment class to store environmental factors.
- DNATokenizer class to tokenize and encode DNA sequences.
- Custom dataset loading and preprocessing.
- Generator and Discriminator models with environmental factors integration.
- Training script for GAN with gradient clipping and learning rate scheduling.
- DNA sequence generation and evaluation.

# Requirements
- Python 3.6 or later
- PyTorch 1.9 or later
- Pandas 1.0 or later
- Matplotlib 3.0 or later
# Installation
Clone the repository and install the required packages:
```
pip install -r requirements.txt
```

# Usage
- Prepare your DNA sequence dataset in a CSV file format. Each line should contain a DNA sequence, and the first line should contain the DNA purpose.

```
*.csv example data:
--
Protein,Purpose,DNA_Sequence
EGFP,Green_Fluorescent_Protein,ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCCTGACCTACGGCGTGCAGTGCTTCAGCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAGGTGAACTTCAAGATCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCAAGCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGTAA
SOD1,Superoxide_Dismutase,MNTEMTSLVKAGTLKKNQGAPTGILQYLGRDSEFVQWFTVNKQTFQYISNKLNSLSNEEIEKELEDFTYKKSGVYALDEAFDRVKKIAEENGVKDVKFFKGLFGSKFESYKAMGKVFQFQEKKEQFDALRAAADMVHGTAPATALYSISLKTPLIQYGGTQYAYCGITAAFTQAIHNTGFAAICHEKIVSDAYKAYRGAQGSLGLATLGVILRGGGIIYQQGTLM
```

- Set the environmental factors you want to consider when generating the DNA sequences.

```
env = Environment(25.0, 7.0, 0.5, 0.0)
```
- Train the GAN using the train_gan() function.

- Generate DNA sequences using the trained GAN.

- Evaluate the generated sequences using the Discriminator model.

# Example output:
```
2025-02-06 14:23:15,632 - INFO - Using device: cuda
2025-02-06 14:23:15,845 - INFO - Starting training...
2025-02-06 14:23:18,234 - INFO - Epoch 1/100 | D loss: 1.8234 | G loss: 2.1456 | D score: 0.3245 | Val G loss: 1.9876 | Val D loss: 1.7654
2025-02-06 14:23:20,567 - INFO - Epoch 2/100 | D loss: 1.6543 | G loss: 1.8765 | D score: 0.4123 | Val G loss: 1.7654 | Val D loss: 1.5432
2025-02-06 14:23:22,890 - INFO - Epoch 3/100 | D loss: 1.4321 | G loss: 1.6543 | D score: 0.4876 | Val G loss: 1.5432 | Val D loss: 1.3210
[...]
2025-02-06 14:45:12,345 - INFO - Epoch 98/100 | D loss: 0.2345 | G loss: 0.3456 | D score: 0.8765 | Val G loss: 0.2123 | Val D loss: 0.1987
2025-02-06 14:45:15,678 - INFO - Epoch 99/100 | D loss: 0.2123 | G loss: 0.3234 | D score: 0.8876 | Val G loss: 0.1987 | Val D loss: 0.1876
2025-02-06 14:45:18,901 - INFO - Epoch 100/100 | D loss: 0.1987 | G loss: 0.3123 | D score: 0.8987 | Val G loss: 0.1876 | Val D loss: 0.1765

2025-02-06 14:45:20,234 - INFO - Generating sample sequences...

2025-02-06 14:45:20,567 - INFO - Generated DNA Sequences Report:
2025-02-06 14:45:20,568 - INFO - ================================

2025-02-06 14:45:20,569 - INFO - DNA Purpose: Promoter sequence for E. coli expression system

2025-02-06 14:45:20,570 - INFO - Generated Sequences:

2025-02-06 14:45:20,571 - INFO - Sequence 1: ATATAATGCCCGTTAGCGATCGAATCGCTAGCTAGCTGATCGATCGAT
2025-02-06 14:45:20,572 - INFO - Discriminator Score: 0.8765
2025-02-06 14:45:20,573 - INFO - GC Content: 48.32%

2025-02-06 14:45:20,574 - INFO - Sequence 2: TTGACAGCTTATCATCGATAAGCTTTAATGCGGTAGTTTATCACAGTT
2025-02-06 14:45:20,575 - INFO - Discriminator Score: 0.8543
2025-02-06 14:45:20,576 - INFO - GC Content: 41.67%

2025-02-06 14:45:20,577 - INFO - Sequence 3: GCTATCGATCGATCGATCGTAGCTAGCTAGCTGATCGATCGTAGCTAG
2025-02-06 14:45:20,578 - INFO - Discriminator Score: 0.8932
2025-02-06 14:45:20,579 - INFO - GC Content: 52.08%

[... continues for all 10 sequences ...]
```

# Potential benefits:

- Data-driven DNA sequence generation: GANs learn to generate DNA sequences from the training data, capturing the underlying distribution of the dataset. This enables the generation of biologically plausible DNA sequences that can survive in specific environments.

- Flexibility: By incorporating environmental factors into the GAN architecture, this method offers flexibility in generating DNA sequences tailored to different environments. Users can adjust the input factors to create sequences that can survive or function optimally under specific conditions.

- Bioengineering and synthetic biology applications: This approach can help researchers in bioengineering and synthetic biology design novel DNA sequences or optimize existing ones for various purposes, such as gene therapy, drug development, or the production of biofuels.

- Understanding environmental adaptation: The generated DNA sequences can provide insights into how organisms adapt to different environmental conditions, which can contribute to the understanding of evolution, ecology, and the mechanisms of environmental adaptation.

- Scalability: GANs can generate a large number of DNA sequences in parallel, making it possible to explore vast regions of sequence space efficiently. This scalability can accelerate the process of designing and testing DNA sequences for various applications.

# Ideas for enhancing the code:
- Improve the architecture: The current generator and discriminator are simple feedforward neural networks. You could explore other architectures like recurrent neural networks (RNNs), long short-term memory (LSTM) networks, or transformers to better capture the inherent sequential nature of DNA sequences.

- Environmental factor representation: The environment is currently represented using a fixed-size vector. You could experiment with other representations, such as embedding layers or even separate encoders, to provide more expressive features for the environment.

- Multi-objective optimization: To optimize for multiple objectives simultaneously, such as maximizing gene expression while minimizing off-target effects, you could modify the loss functions or employ multi-objective optimization techniques.

- Data augmentation: To increase the variety of training data, you could apply data augmentation techniques on the input DNA sequences, such as generating reverse complements, simulating point mutations, or introducing random gaps.

- Hyperparameter tuning: Experiment with different values for the model's hyperparameters, such as learning rate, number of layers, or hidden dimensions, to find the optimal configuration for generating high-quality DNA sequences.

- Conditional GANs (cGANs): Instead of incorporating environmental factors directly into the generator and discriminator, you could implement a conditional GAN, which conditions the generation process on additional information, such as the environment or specific DNA sequence properties.

- Incorporate domain knowledge: Integrate biological domain knowledge or constraints, such as codon usage bias, functional elements, or sequence motifs, into the model to ensure that generated sequences are biologically relevant and functional.

- Evaluate performance using real-world metrics: Develop methods to evaluate the performance of the generated sequences using real-world metrics or experimental data, such as gene expression levels, protein stability, or metabolic pathway efficiency.

- Don't be lazy, do it!.. cbrwx
