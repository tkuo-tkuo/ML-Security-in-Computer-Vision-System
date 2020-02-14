import torch, os, math, random, argparse, RBF_utils
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np

class reduced_model(nn.Module):
    def __init__(self, num_of_elements):
        super(reduced_model, self).__init__()
        self.fc = nn.Linear(num_of_elements, 10)
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        return self.softmax(self.fc(x))

def generate_reduced_model(signatures, Y, model_name):
    num_of_elements = signatures.shape[1]
    r_model = reduced_model(num_of_elements) 
    loss_func, optimizer = nn.CrossEntropyLoss(), torch.optim.Adam(r_model.parameters(), lr=1e-3)

    # training 
    epoches = 5
    for _ in range(epoches):
        total_loss = None
        for (x, y) in zip(signatures, Y):
            # Transform from numpy to torch & correct the shape (expand dimension) and type (float32 and int64)
            data = torch.from_numpy(np.expand_dims(x, axis=0).astype(np.float32))
            label = torch.from_numpy(np.array([y]).astype(np.int64))

            # Forwarding
            outputs = r_model.forward(data)
            loss = loss_func(outputs, label)

            if total_loss is None: total_loss = loss
            else: total_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(total_loss)

        # test acc           
        outputs = r_model.forward(signatures)
        predictions = torch.argmax(outputs, dim=1)
        correct = (predictions == Y).sum()
        print(model_name, 'acc:', correct.item()/signatures.shape[0])

    torch.save(r_model, model_name)

def generate__reduced_signatures_on_reduced_model(signatures, model_name):
    r_model = torch.load(model_name)
    outputs = r_model.forward(signatures)
    return outputs



torch.manual_seed(777)
class RbfNet(nn.Module):
    def __init__(self, centers, num_class=10):
        super(RbfNet, self).__init__()
        self.num_class = num_class
        self.num_centers = centers.size(0)

        self.centers = nn.Parameter(centers)
        self.beta = nn.Parameter(torch.ones(1,self.num_centers)/10)
        self.linear = nn.Linear(self.num_centers, self.num_class, bias=True)
        self.softmax = nn.Softmax()
        RBF_utils.initialize_weights(self)

    def kernel_fun(self, batches):
        n_input = batches.size(0) # number of inputs
        A = self.centers.view(self.num_centers,-1).repeat(n_input,1,1)
        B = batches.view(n_input,-1).unsqueeze(1).repeat(1,self.num_centers,1)
        C = torch.exp(-self.beta.mul((A-B).pow(2).sum(2,keepdim=False).sqrt() ) )
        return C

    def forward(self, batches):
        radial_val = self.kernel_fun(batches)
        class_score = self.linear(radial_val)
        return self.softmax(class_score)


class RBFN(object):
    def __init__(self, args, train_X, train_Y, test_X, test_Y):
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y

        self.max_epoch = args['epoch']
        self.batch_size = args['batch_size']
        self.save_dir = args['save_dir']
        self.dataset = 'MNIST'
        self.model_name = args['model_name']
        self.lr = args['lr']
        self.num_class = args['num_class']
        self.num_centers = args['num_centers']
        self.num_of_elements = args['num_of_elements']

        self.train_data = datasets.MNIST(root = 'data/',
                                        train = True,
                                        transform = transforms.ToTensor(),
                                        download = True)
        self.test_data = datasets.MNIST(root = 'data/',
                                    train = False,
                                    transform = transforms.ToTensor(),
                                    download = True)

        self.data_loader = DataLoader(dataset = self.train_data,
                                        batch_size = self.batch_size,
                                        shuffle = True,
                                        num_workers = 1,
                                        pin_memory=True)

        # self.center_id = random.sample(range(1,len(self.train_X)), self.num_centers)
        #TODO: sometime nan value exists in centers (b!=b).nonzero()
        #self.centers = self.train_data.train_data[self.center_id,].unsqueeze(1).float().div(255)
        self.centers = torch.rand(self.num_centers,self.num_of_elements)

        self.model = RbfNet(self.centers, num_class=self.num_class)
        # self.model.cuda()
        # RBF_utils.print_network(self.model)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fun = nn.CrossEntropyLoss()

    def train(self):
        self.model.train()
        for epoch in range(self.max_epoch):
            avg_cost = 0
            total_batch = math.ceil(len(self.train_X)/self.batch_size)
            

            for i in range(total_batch):
                lower_index, upper_index = i*self.batch_size, (i+1)*self.batch_size
                if upper_index >= len(self.train_X): break 
                
                X = self.train_X[lower_index:upper_index]
                Y = self.train_Y[lower_index:upper_index]

                # X = Variable(batch_images.view(-1, 28 * 28)).cuda()
                # Y = Variable(batch_labels).cuda()        # label is not one-hot encoded
                X = Variable(X)
                Y = Variable(Y)             # label is not one-hot encoded

                # import ipdb; ipdb.set_trace(context=20)

                self.optimizer.zero_grad()             # Zero Gradient Container
                Y_prediction = self.model(X)           # Forward Propagation
                cost = self.loss_fun(Y_prediction, Y) # compute cost
                cost.backward()                   # compute gradient
                self.optimizer.step()                  # gradient update

                # import ipdb; ipdb.set_trace(context=20)
                avg_cost += cost / total_batch
                # print("center sum: %f" % (self.model.centers.data.sum()))

            print("[Epoch: {:>4}] cost = {:>.9}".format(epoch + 1, avg_cost.data.item()))

        print(" [*] Training finished!")

    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        total_confidence = 0
        
        for i in range(len(self.test_X)):
            X = self.test_X[i].detach().numpy()
            Y = self.test_Y[i]

            # images = Variable(images.view(-1, 28*28)).cuda()
            X = torch.from_numpy(np.expand_dims(X, axis=0).astype(np.float32))
            X = Variable(X)
            outputs = self.model(X)
            _, predicted = torch.max(outputs.data, 1)
            total_confidence += outputs[0][Y]
            
            total += 1
            correct += (predicted == Y).sum()

        print('Accuracy of the network on the '+str(total)+' test images: %f %%' % (100 * correct.item() / total))
        print('Average confidence of the network on the '+str(total)+' test images', total_confidence.item()/total)
        print(" [*] Testing finished!")

    def eval(self, x, y):
        self.model.eval()

        X = x.detach().numpy()
        Y = y

        X = torch.from_numpy(np.expand_dims(X, axis=0).astype(np.float32))
        X = Variable(X)
        outputs = self.model(X)[0]
        confidence = outputs[y]
        return confidence.item()

    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.model.state_dict(), os.path.join(save_dir, self.model_name+'.pkl'))
        print(" [*] Done saving check point yo!")

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset)
        self.model.load_state_dict(torch.load(os.path.join(save_dir, self.model_name+'.pkl')))
        # print(" [*] Done weight loading!")

def main():
    parser = argparse.ArgumentParser(description='Radial Based Network')
    parser.add_argument('--lr', default = 0.01, type=float, help='learning rate')
    parser.add_argument('--batch_size', default = 200, type=int, help='batch size')
    parser.add_argument('--epoch', default = 5, type=int, help='epoch size')
    parser.add_argument('--num_class', default = 10, type=int, help='num labels')
    parser.add_argument('--num_centers', default=300, type=int, help='num centers')
    parser.add_argument('--save_dir', default = 'ckpoints', type=str, help='ckpoint loc')
    parser.add_argument('--result_dir', default = 'outs', type=str, help='output')
    parser.add_argument('--dataset', default = 'MNIST', type=str )
    parser.add_argument('--model_name', default='RBFN', type=str )
    parser.add_argument('--cuda', default=False, type=bool )
    args = parser.parse_args()

    rbfn = RBFN(args)
    rbfn.train()
    rbfn.save()
    rbfn.load()
    rbfn.test()

    return 0


if __name__ == "__main__":
    main()