'''
Convert .pth weight to .txt format for fixed-point quantization
'''
import os.path

import torch
import numpy as np
from Model.Student import SCNN
from Model.Teacher import TCNN


if __name__ == '__main__':
    model =SCNN()
    #model  = TCNN()
    file = "Pth/" + "SCNN__-6_0HP_0.32" + ".pth"

    #file = "Pth/" + "DKD11_Tmodel-6_0HP_1.0" + ".pth"
    model.load_state_dict(torch.load(file))


    print(model)
    for name, param in model.named_parameters():
        print(f'{name}max:{param.max()}' )
        print(f'{name}min:{param.min()}' )

    if not os.path.exists('Weight_Parameters/Float'):
        os.mkdir('Weight_Parameters/Float')
    if not os.path.exists('Weight_Parameters/Fixed_Point'):
        os.mkdir('Weight_Parameters/Fixed_Point')
    # save weight parameters to .txt
    for i, (name, param) in enumerate(model.named_parameters()):
        data = torch.tensor(param.detach().cpu().numpy().astype('float32'))
        with open(f'Weight_Parameters/Float/scnn_layer_{i}.txt', 'w') as f:
            # f.write(name + '\n')
            f.write('\n'.join([str(x.item()) for x in data.view(-1)]) + '\n')

