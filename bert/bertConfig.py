def bert_opt(opt):
    #BERT parameter
    opt.bert_path = "/data/pengyu/bert-base-uncased"
    opt.bert_batch_size = 4
    opt.bert_learning_rate = 1e-5
    opt.bert_adam_epsilon = 1e-8
    opt.bert_dropout = 0.5
    opt.ntg_learning_rate = 0.001
    
    return opt     