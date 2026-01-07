### Assignment 1 summary
This notebook implements two different image classification tasks using PyTorch: Transfer Learning on CIFAR-10 and a Custom MLP on MNIST.
1. Transfer Learning with DenseNet121 on CIFAR-10
     - Dataset: CIFAR-10 (10 classes of colored images).
     - Preprocessing:
         - Images resized to 224x224.
         - Data augmentation applied to training set: Random Rotation (15 degrees) and Random Horizontal Flip.
         - Normalization using ImageNet mean and standard deviation.
     - Model Architecture:
         - Used a pre-trained DenseNet121 model (IMAGENET1K_V1 weights).
         - Transfer Learning Strategy: Frozen the feature extraction layers (model.features) and replaced the final classifier layer to output 10 classes.
     - Training:
         - Optimizer: Adam (lr=0.001).
         - Loss Function: CrossEntropyLoss.
         - Epochs: 5.
     - Evaluation:
         - Achieved a test accuracy of approximately 81.12%.
         - Visualized predictions and generated a Confusion Matrix using Seaborn.
2. Custom Multi-Layer Perceptron (MLP) on MNIST
     - Dataset: MNIST (Handwritten digits, grayscale).
     - Preprocessing:
         - Converted to Tensor and normalized using MNIST mean (0.1307) and std (0.3081).
     - Model Architecture:
         - Built a custom fully connected neural network (nn.Linear).
         - Structure:
             - Input Layer: Flattened 28x28 image.
             - Hidden Layer 1: 128 units + ReLU + Dropout (0.2).
             - Hidden Layer 2: 64 units + ReLU + Dropout (0.5).
             - Output Layer: 10 units.
     - Training:
         - Optimizer: Adam (lr=0.001).
         - Loss Function: CrossEntropyLoss.
         - Epochs: 20.
     - Evaluation:
         - Achieved a test accuracy of approximately 97.83%.
         - Visualized individual predictions and plotted a Confusion Matrix.
      
### Assignment 2 summary
This notebook implements two sentiment analysis pipelines: a classical approach using Word Embeddings on tweets and a modern Deep Learning approach using Transformers on movie reviews.
1. Sentiment Analysis on Airline Tweets (Word2Vec + Logistic Regression)
     - Dataset: Airline Tweets (tweets.csv).
     - Preprocessing:
         - Implemented a robust preprocess_text function using NLTK and Regex.
         - Steps included: Lowercasing, expanding contractions, removing URLs/Mentions/Hashtags/Emojis, Tokenization, and Lemmatization.
     - Feature Extraction:
         - Used the pre-trained Google News Word2Vec model (300 dimensions).
         - Converted tweets into fixed-length vectors by averaging the word vectors of the tokens in the tweet.
     - Model:
         - Mapped targets to 3 classes: Positive (1), Neutral (0), Negative (-1).
         - Trained a Logistic Regression classifier.
     - Evaluation:
         - Split data into 80% train / 20% test.
         - Reported classification accuracy on the test set.
      
2. Sentiment Analysis on IMDB Reviews (Fine-tuning BERT)
     - Dataset: IMDB Movie Reviews (loaded via Hugging Face datasets).
     - Preprocessing:
         - Used BertTokenizer (bert-base-uncased).
         - Tokenized text with padding and truncation to a max length of 512.
     - Model Architecture:
         - Fine-tuned a pre-trained BERT model (BertForSequenceClassification) for binary classification.
     - Training:
         - Used the Hugging Face Trainer API.
         - Hyperparameters: 3 epochs, batch size of 8, weight decay 0.01, warmup steps 500.
         - Utilized GPU acceleration (CUDA).
     - Evaluation:
         - Calculated Accuracy and F1-Score on the test set.
         - Saved the fine-tuned model and demonstrated

### Assignment 3 summary
1. Explainable ai
      - Dataset: CIFAR-10 (10 classes).
      - Preprocessing:
          - Resized images to 224x224 to match ResNet's expected input.
          - Data augmentation (Random Horizontal Flip, Random Rotation).
          - Normalization using ImageNet statistics.
      - Model Architecture:
          - Loaded a pre-trained ResNet-18 (IMAGENET1K_V1 weights).
          - Fine-tuning Strategy: Froze all layers initially, then unfroze layer4 and the final fully connected layer (fc). Replaced the fc layer to output 10 classes.
      - Training:
          - Optimizer: Adam (lr=0.001).
          - Loss Function: CrossEntropyLoss.
          - Scheduler: StepLR (step_size=5, gamma=0.1).
          - Epochs: 10.
      - Explainability techniques used:
          - Convolutional Filter Visualization
          - Feature Map (Activation) Visualization
          - Misclassification Analysis
          - Grad-CAM      
        
### Final Project summary
This project implements an end-to-end deep learning model capable of generating descriptive captions for images. It utilizes a ResNet-18 backbone for visual feature extraction and a Transformer Decoder for text generation, trained on the Flickr8k dataset.
1. Dataset (Flickr8k)
   - Contains 8,000 images, each with 5 different captions.
   - Preprocessing: Images were resized to 224x224 and normalized using ImageNet standards.
   - Splits: Data was separated into Training (6,000), Validation (1,000), and Test (1,000) sets.
   - Vocabulary: Built a vocabulary of ~3,000 words (frequency threshold = 1) including special tokens (<bos>, <eos>, <pad>, <unk>).
2. Model Architecture
   - Encoder (CNN)
      - Backbone: ResNet-18 (Pretrained on ImageNet).
      - Function: Extracts high-level visual features from input images.
      - Optimization Strategies:
         - Feature Caching: Pre-computed properties for faster initial training.
         - Fine-Tuning: Unfrozen the last convolutional block to learn task-specific features during the end-to-end training phase.
   - Decoder(Transformer)
      - Architecture: PyTorch nn.TransformerDecoder.
      - Configuration: 4 decoder layers, 8 attention heads, 512 embedding dimension (d_model).
      - Mechanism: Uses Self-Attention to process text and Cross-Attention to attend to image features extracted by the CNN.
      - Input: Positional encodings + Word embeddings.
 3. Implementation details
   - Data Processing: Tokenization, vocabulary building, and custom PyTorch Dataset creation.
   - Feature Caching: Implemented a mode to pre-calculate and save CNN features to .pt files to speed up decoder prototyping.
   - End-to-End Training: Built a full pipeline where images are loaded directly, allowing gradients to flow back into the CNN.
   - Differential Learning Rates: Used a lower learning rate (1e-5) for the pre-trained CNN and a higher learning rate (2e-4) for the Transformer Decoder to preserve visual knowledge while learning syntax.
   - Training Configurations:
      - Optimizer: Adam
      - Loss Function: Cross Entropy Loss (ignoring padding)
      - Scheduler: StepLR (learning rate decay)
      - Epochs: 20
      - Regularization: Dropout (0.1) and Weight Decay.
 4. Evaluation & Results: The model was evaluated using standard captioning metrics and qualitative analysis
   - Metrics:
      - BLEU-1 to BLEU-4: Measured n-gram overlap with ground truth.
      - METEOR: Evaluated alignment between hypothesis and reference.
      - Repetition Rate: Analyzed degenerate repetition to ensure diverse generation.
 5. Explainability: Implemented Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize which parts of the image the model focused on when generating captions (specifically targeting the End-of-Sentence token).
   - Success Cases: Model focuses on relevant primary subjects (e.g., dogs, people).
   - Failure Analysis: Visualized cases where the model looked at background noise or misclassified objects due to domain shift or occlusion.    
