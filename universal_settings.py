

n_base_classes = {
    'omniglot':4112, # val:688, novel:1692
    'CUB':100, # val:50, novel:50
    'miniImagenet':64, # val:16, novel:20
    'cross_char':1597, 'cross_char_half':758, 'cross_char_quarter':350, # val:31, novel:31
    'cross_char_base12lang':350, 'cross_char_base3lang':69, 'cross_char_base1lang':20, # val:31, novel:31
    'cross':100, 'cross_base80cl':80, 'cross_base20cl':20, # val:50, novel:50
}

stop_epoch = {
    'baseline':100, 
    'protonet':1000, 
    'relationnet':1000, 
}

patience = {
    'baseline':10, 
    'protonet':70, 
    'relationnet':70, 
}

should_aug_datasets = [
    'CUB', 'miniImagenet', 
    'cross', 'cross_base80cl', 'cross_base40cl', 'cross_base20cl'
]
