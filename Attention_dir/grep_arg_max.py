import torch
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--heads', type=int, default=50)
args = parser.parse_args()
his_data = []
for i in range(args.heads):
    atten_array = torch.load('dblp_attention_%d.pt'%(i)).squeeze(dim=2)
    tbp = atten_array.cpu().detach().numpy()
    print(np.argmax(tbp, axis=1))
    his_data = his_data + list(np.argmax(tbp, axis=1))
his_data = np.array(his_data)
num_of_heads = len(his_data)*1.0
print(his_data)
import matplotlib.pyplot as plt
#n, bins, patches = plt.hist(his_data,density=True,facecolor='g',bins=[0,1,2,3,4,5,6])
y_pos = np.arange(1,7)
height = [sum(his_data==1)/num_of_heads,sum(his_data==2)/num_of_heads,sum(his_data==3)/num_of_heads,\
        sum(his_data==4)/num_of_heads,sum(his_data==5)/num_of_heads,sum(his_data==6)/num_of_heads]
plt.bar(y_pos, height)
y_pos = np.arange(1,7)
plt.xlim(0,7)
plt.xticks(y_pos, [r'$A$', r'$A^2$', r'$A^3$', r'$\Psi_1$', r'$\Psi_2$',r'$\Psi_3$'])
plt.rcParams["font.weight"] = "bold"

plt.savefig('dblp_argmax.pdf')
