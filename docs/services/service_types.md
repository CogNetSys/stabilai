# Machine Learning Services



### **1. DataForgeML: Tabular and Dense Models Service**
- **Includes**:
     - Multilayer Perceptrons (MLPs)
     - Autoencoders (AEs)
     - Variational Autoencoders (VAEs)
     - Sparse Neural Networks
- **Data Format**: Tabular or vector data (`[batch_size, feature_dim]`).
- **Purpose**:
     - Ideal for processing structured, feature-based data.
     - Supports reconstruction loss, KL-divergence, and dense optimization needs.

---

### **2. TempoTune: Sequential Models Service**
- **Includes**:
     - Recurrent Neural Networks (RNNs)
     - Transformers (BERT, GPT, etc.)
     - Attention-Based Models
    - Recursive Neural Networks
     - Liquid Neural Networks
- **Data Format**: Sequential or temporal data (`[batch_size, sequence_length, feature_dim]`).
- **Purpose**:
     - Designed to master the rhythm and flow of sequential data, such as time-series and natural language.
     - Provides advanced tokenization, sequence alignment, and attention mechanisms for seamless data processing.
     - Optimized for capturing temporal patterns and contextual relationships in sequential datasets.

---

### **3. VisionVista: Vision Models Service**
- **Includes**:
     - Convolutional Neural Networks (CNNs)
     - Capsule Networks
     - Diffusion Models
     - Generative Adversarial Networks (GANs)
- **Data Format**: Image tensors (`[batch_size, channels, height, width]`).
- **Purpose**:
     - Specializes in 2D spatial data such as images.
    - Offers advanced data augmentation and convolutional architectures.

---

### **4. GraphGenius: Graph-Based Models Service**
- **Includes**:
     - Graph Neural Networks (GNNs)
     - Graph Attention Networks (GATs)
     - Physics-Informed Neural Networks (PINNs)
- **Data Format**: Graph data (node features, edge features, adjacency matrices, or `edge_index` format).
- **Purpose**:
     - Designed for graph-structured data and preprocessing pipelines.
     - Leverages tools like PyTorch Geometric for streamlined implementation.

---

### **5. IntelliSphere: Specialized Models Service**
- **Includes**:
     - Reinforcement Learning Models
     - Bayesian Networks
     - Gaussian Processes (GPs)
     - Spiking Neural Networks (SNNs)
     - Self-Organizing Maps (SOMs)
- **Data Format**:
     - **Reinforcement Learning**: State-action-reward tuples.
     - **Bayesian Networks & GPs**: Probabilistic models.
     - **SNNs & SOMs**: Specialized data formats for neural coding or clustering.
- **Purpose**:
     - Provides specialized services for RL environments, probabilistic sampling, and neural clustering.

---

## Service Groups Summary

1. **DataForge**: Tabular and dense models like MLPs and Autoencoders.
2. **TempoTune**: Sequential models such as RNNs, Transformers, and Attention-based models.
3. **VisionVista**: Vision-based models including CNNs, GANs, and Diffusion Models.
4. **GraphGenius**: Graph-focused models like GNNs and GATs.
5. **IntelliSphere**: Specialized and task-specific models such as RL and Bayesian networks.

---