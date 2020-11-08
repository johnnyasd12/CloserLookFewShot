

n_base_classes = {
    'omniglot':4112, 'omniglot_400cl':400, 'omniglot_40cl':40, # val:688, novel:1692
    'CUB':100, 'CUB_base50cl':50, 'CUB_base25cl':25, # val:50, novel:50
    'miniImagenet':64, # val:16, novel:20
    'cross_char':1597, 'cross_char_half':758, 'cross_char_quarter':350, # val:31, novel:31
    'cross_char_base12lang':350, 'cross_char_base3lang':69, 'cross_char_base1lang':20, # val:31, novel:31
    'cross_char2':1597, 'cross_char2_quarter':350, 'cross_char2_base3lang':69, 'cross_char2_base1lang':20, # val:31, novel:31
    'cross':100, 'cross_base80cl':80, 'cross_base40cl':40, 'cross_base20cl':20, # val:50, novel:50
    'try_virtual_20info_base200cl_50info':200, 
    'virtual_info0029_base_info0029':400, 'virtual_info0029_base_info1039':400, 'virtual_info0029_base_info2049':400, 'virtual_info0029_base_info3059':400, 
}

stop_epoch = {
    'baseline':500, 
    'protonet':1000, 
#     'relationnet':1000, 
    'relationnet_softmax':1000, 
    'matchingnet':1000, 
}

patience = {
    'baseline':10, 
    'protonet':70, 
#     'relationnet':70, 
    'relationnet_softmax':70, 
    'matchingnet':70, 
}

should_aug_datasets = [
    'CUB', 'CUB_base50cl', 'CUB_base25cl', 
    'miniImagenet', 
    'cross', 'cross_base80cl', 'cross_base40cl', 'cross_base20cl'
]
