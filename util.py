from copy import deepcopy
from tqdm import tqdm

import torch

def train_rbm(rbm, train_loader, args, validation_loader = None, method = 'discriminative', generative_factor = None, discriminative_factor = 1):
    print('Training RBM...')
    loss_list = []
    best_validation_acc = 0
    best_validation_model = None
    patience_count = 0

    for epoch in range(args.epochs):
        epoch_error = 0.0
        for batch, labels in tqdm(train_loader, total = len(train_loader)):
            batch = batch.view(len(batch), args.visible_units)  # flatten input data

            if args.cuda:
                batch = batch.float().cuda()
                labels = labels.cuda()
                
            
            if method == 'generative':
                batch_error, predicted_labels = rbm.contrastive_divergence(batch, labels)
            elif method == 'discriminative':
                batch_error, _ = rbm.discriminative_training(batch, labels)
            elif method == 'hybrid' and generative_factor and discriminative_factor:
                batch_error, _ = rbm.hybrid_training(batch, labels, generative_factor, discriminative_factor)
            else:
                raise NotImplementedError
            epoch_error += batch_error

        if validation_loader:
            validation_acc = test_rbm_model(rbm, validation_loader, args)
            if validation_acc > best_validation_acc:
                best_validation_acc = validation_acc
                best_validation_model = deepcopy(rbm)
                patience_count = 0
            else:
                patience_count += 1

            if patience_count == args.early_stop:
                return best_validation_model
            print("Validation Accuracy %.4f" % (validation_acc))
        else:
            best_validation_model = rbm
                
        loss_list.append(epoch_error/len(train_loader))
        
        print('Epoch Error (epoch=%d): %.4f' % (epoch, epoch_error/len(train_loader)))
        
    return loss_list, best_validation_model
        
        
        
def test_rbm_model(rbm_model, test_loader, args):
    correct = 0
    total = 0
    
    for batch, labels in tqdm(test_loader):
        batch = batch.view(len(batch), args.visible_units)  # flatten input data

        if args.cuda:
            batch = batch.float().cuda()
            labels = labels.cuda()

        predicted_probabilities = rbm_model.sample_class_given_x(batch)

        _ , predicted =  torch.max(predicted_probabilities, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
     
      
    print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
    
    return correct/total