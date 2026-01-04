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
