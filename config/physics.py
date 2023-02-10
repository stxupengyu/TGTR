def physics_opt(opt):
    #multi task parameter
    opt.delta = 1

    #topic parameter
    opt.topic_num = 60

    #model parameter
    opt.linear_size = [512,256]

    #training parameter
    opt.gpuid = 3
    opt.learning_rate = 1e-3
    return opt  
