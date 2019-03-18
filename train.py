import argparse
import myutils
parser = argparse.ArgumentParser()
parser.add_argument('data_directory', action='store')
parser.add_argument('--gpu', action='store', default='gpu')
parser.add_argument('--save_dir', dest='save_dir', action='store', default = 'checkpoint.pth')
parser.add_argument('--arch', dest="arch", action="store", default="vgg19", type=str)
parser.add_argument('--learning_rate', action='store', default = 0.001, dest='learning_rate')
parser.add_argument('--hidden_units', action='store', default = 1024,  dest='hidden_units')
parser.add_argument('--epochs', action='store',default = 7, dest='epochs')


results = parser.parse_args()

print('data_directory     = {!r}'.format(results.data_directory))
print('gpu     = {!r}'.format(results.gpu))

data_dir = results.data_directory
save_dir = results.save_dir
learning_rate = results.learning_rate
arch = results.arch
hidden_units = results.hidden_units
epochs = results.epochs
gpu = results.gpu

trainloader, validloader, testloader, train_data  = myutils.load_data(data_dir)
model, criterion, optimizer = myutils.build_the_model(arch, hidden_units,learning_rate,gpu)
model, optimizer = myutils.train_model(model, epochs,trainloader, validloader, criterion, optimizer,gpu)
#Save it
myutils.save_model(model, train_data, optimizer, save_dir, arch, epochs)
print('Done Training the model')