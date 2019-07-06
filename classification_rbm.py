import torch

class ClassificationRBM():

    def __init__(self, num_visible, num_hidden, k, num_classes = 10, learning_rate=0.05, sparse_constant = 0.00, use_cuda=True):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.learning_rate = learning_rate
        self.use_cuda = use_cuda
        self.num_classes = num_classes
        self.loss = torch.nn.CrossEntropyLoss()
        self.sparse_constant = sparse_constant

        self.weights = torch.randn(num_visible, num_hidden, ) * 0.1
        self.visible_bias = torch.ones(num_visible) * 0.5
        self.hidden_bias = torch.zeros(num_hidden)
        self.class_bias = torch.zeros(num_classes) 
        self.class_weights = torch.zeros(num_classes, num_hidden) * 0.5


        if self.use_cuda:
            self.weights = self.weights.cuda()
            self.visible_bias = self.visible_bias.cuda()
            self.hidden_bias = self.hidden_bias.cuda()
            self.class_weights = self.class_weights.cuda()
            self.class_bias = self.class_bias.cuda()
       

    def sample_hidden(self, visible_activations, class_activations):
        hidden_activations = torch.matmul(visible_activations, self.weights) + self.hidden_bias + torch.matmul(class_activations, self.class_weights)
        hidden_probabilities = self._sigmoid(hidden_activations)
        return hidden_probabilities

    def sample_visible(self, hidden_activations):
        visible_activations = torch.matmul(hidden_activations, self.weights.t()) + self.visible_bias
        visible_probabilities = self._sigmoid(visible_activations)
        return visible_probabilities
    
    def sample_class(self, hidden_activations):
        
        class_probablities = torch.exp(torch.matmul(hidden_activations,self.class_weights.t()) + self.class_bias)
        #print(torch.sum(class_probablities, dim = 1).shape)
        class_probablities = torch.nn.functional.normalize(class_probablities, p = 1, dim = 1)
        #print(class_probablities.shape)
        return class_probablities
    
    def sample_class_given_x(self, input_data):
        """Sampling the label given input data in time O(num_hidden * num_visible + num_classes * num_classes) for each example"""
        
    
        precomputed_factor = torch.matmul(input_data, self.weights) + self.hidden_bias
        class_probabilities = torch.zeros((input_data.shape[0], self.num_classes)).cuda()

        for y in range(self.num_classes):
            prod = torch.zeros(input_data.shape[0], device = input_data.device)
            prod += self.class_bias[y]
            for j in range(self.num_hidden):
                prod += torch.log(1 + torch.exp(precomputed_factor[:,j] + self.class_weights[y, j]))
            #print(prod)
            class_probabilities[:, y] = prod  

        copy_probabilities = torch.zeros(class_probabilities.shape, device = input_data.device)

        for c in range(self.num_classes):
          for d in range(self.num_classes):
            copy_probabilities[:, c] += torch.exp(-1*class_probabilities[:, c] + class_probabilities[:, d])

        copy_probabilities = 1/copy_probabilities


        class_probabilities = copy_probabilities

        return class_probabilities
   
 
    def contrastive_divergence(self, input_data, class_label, factor = 1):

        class_one_hot = torch.nn.functional.one_hot(class_label, num_classes = self.num_classes).float()
        # Positive phase
        positive_hidden_probabilities = self.sample_hidden(input_data, class_one_hot)
        positive_hidden_activations = (positive_hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()
        positive_associations = torch.matmul(input_data.t(), positive_hidden_probabilities)
        positive_class_associations = torch.matmul(class_one_hot.t(), positive_hidden_probabilities)
        positive_class_probabilities = self.sample_class(positive_hidden_activations)
        # Negative phase
        hidden_activations = positive_hidden_activations

        for step in range(self.k):
            visible_probabilities = self.sample_visible(hidden_activations)
            visible_activations = (visible_probabilities >= self._random_probabilities(self.num_visible)).float()
            
            class_probabilities = self.sample_class(hidden_activations)
            class_activations = torch.nn.functional.one_hot(torch.argmax(class_probabilities, dim = 1), num_classes = 10).float()
            
            hidden_probabilities = self.sample_hidden(visible_activations, class_activations)
            hidden_activations = (hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()

        negative_visible_activations = visible_activations
        negative_hidden_probabilities = hidden_probabilities
        negetive_class_activations = class_activations
        #print(negative_visible_activations.shape, negative_hidden_probabilities.shape, negetive_class_activations.shape)
        negative_associations = torch.matmul(negative_visible_activations.t(), negative_hidden_probabilities)
        negetive_class_associations = torch.matmul(negetive_class_activations.t(), negative_hidden_probabilities)
        # Update parameters
    
        self.weights_grad = (positive_associations - negative_associations)

        self.class_weights_grad = (positive_class_associations - negetive_class_associations)
        
        self.visible_bias_grad = torch.sum(input_data - negative_visible_activations, dim=0)
        
        self.hidden_bias_grad = torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim=0)
        
        self.class_bias_grad = torch.sum(class_one_hot - negetive_class_activations, dim = 0)	

        batch_size = input_data.size(0)

        self.update_weights(batch_size, factor)
        
        class_probabilities = self.sample_class_given_x(input_data)
        # Compute reconstruction error
        error = self.loss(class_probabilities, class_label)
        
        _ , predicted = torch.max(class_probabilities, 1) 

        return error, predicted
      
      
    def update_weights(self, batch_size, factor = 1):
      
        self.weights += factor*self.weights_grad * self.learning_rate / batch_size
        self.visible_bias += factor*self.visible_bias_grad * self.learning_rate / batch_size
        self.hidden_bias += factor*self.hidden_bias_grad * self.learning_rate / batch_size
        self.class_bias += factor*self.class_bias_grad * self.learning_rate / batch_size
        self.class_weights += factor*self.class_weights_grad * self.learning_rate / batch_size
        
        
        self.visible_bias -= self.sparse_constant
          
        self.hidden_bias -= self.sparse_constant
        self.class_bias -= self.sparse_constant

      
    def discriminative_training(self, input_data, class_label, factor = 1):
      
        batch_size = input_data.size(0)
        
        class_one_hot = torch.nn.functional.one_hot(class_label, num_classes = self.num_classes).float()
        o_y_j =  self._sigmoid((torch.matmul(input_data, self.weights)+ self.hidden_bias).unsqueeze_(-1).expand(-1, -1, self.num_classes) + self.class_weights.t())
        #print(o_y_j)
        class_probabilities = self.sample_class_given_x(input_data)

        positive_sum = torch.zeros(batch_size, self.num_hidden, device = input_data.device)
        class_weight_grad = torch.zeros(self.num_classes, self.num_hidden, device = input_data.device)

        for i,c in enumerate(class_label):
            positive_sum[i] += o_y_j[i, : , c]
            class_weight_grad[c ,:] += positive_sum[i]

        #print(positive_sum)
        unfolded_input = input_data.unsqueeze(-1).expand(-1, -1, self.num_hidden)
        positive_associations = torch.sum(torch.mul(unfolded_input, positive_sum.unsqueeze_(1)), dim = 0)
        #print(positive_associations.shape)

        negetive_sum  = torch.zeros(batch_size, self.num_hidden, device = input_data.device)

        for c in range(self.num_classes):
          class_weight_grad[c, :] -= torch.sum(o_y_j[:,:,c] * class_probabilities[:,c].unsqueeze_(-1), dim = 0)  
          negetive_sum += o_y_j[:,:,c] * class_probabilities[:,c].unsqueeze(-1)

        negetive_associations = torch.sum(torch.mul(unfolded_input, negetive_sum.unsqueeze_(1)), dim = 0)

        self.weights_grad = (positive_associations - negetive_associations)

        self.class_weights_grad = (class_weight_grad)

        self.hidden_bias_grad = torch.sum(positive_sum.squeeze_() - negetive_sum.squeeze_(), dim=0)

        self.class_bias_grad = torch.sum(class_one_hot - class_probabilities, dim = 0)	
        
        self.visible_bias_grad = 0

        self.update_weights(batch_size, factor)

        error = self.loss(class_probabilities, class_label)
        
        _ , predicted = torch.max(class_probabilities, 1) 
        

        return error, predicted
      
    def hybrid_training(self, input_data, class_label, generative_factor, discriminative_factor = 1):
      
      class_probabilities = self.sample_class_given_x(input_data)
      
      self.contrastive_divergence(input_data, class_label, generative_factor)
      self.discriminative_training(input_data, class_label, discriminative_factor)
      
      error = self.loss(class_probabilities, class_label)
      
      _ , predicted = torch.max(class_probabilities, 1) 
        

      return error, predicted
      
        
    def _sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def _random_probabilities(self, num):
        random_probabilities = torch.rand(num)

        if self.use_cuda:
            random_probabilities = random_probabilities.cuda()

        return random_probabilities



