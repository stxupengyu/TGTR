def law_opt(opt):
    #multi task parameter
    opt.delta = 0.01

    #topic parameter
    opt.topic_num = 90

    #model parameter
    opt.linear_size = [256,256]#[256,256]

    #training parameter
    opt.gpuid = 3
    opt.learning_rate = 2e-3
    return opt  
