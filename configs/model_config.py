from model.z2q_128x128 import Z2Q_128x128
from model.z2v_128x128 import Z128x128
from model.z2q_16x128 import Z2Q_16x128
from model.add2VAqFE import BiasAddModel
from model.base_model import Base
from model_adapter.vq_cmpd_adapter import VQCmpdAdapter

from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from model_adapter.base_adapter import BaseModelAdapter

def get_model(configs, device, dataset, train_loader, writer):
    tsgv_model_name, generator_name, adapter_name = configs.model_name.split('_')
    if tsgv_model_name == 'base':
        model = Base(configs=configs, word_vectors=dataset['word_vector']).to(device)
    elif tsgv_model_name == 'add2vaqfe':
        model = BiasAddModel(configs=configs, word_vectors=dataset['word_vector']).to(device)
    else:
        raise Exception(f"tsgv_model_name error: {tsgv_model_name}")

    if generator_name == 'none':
        generator = None
    elif generator_name == 'vqc128':
        generator = []
        if configs.task == 'activitynet':
            generator.append(Z128x128().to(device))
            generator.append(Z2Q_128x128().to(device))
        else:
            generator.append(Z128x128().to(device))
            generator.append(Z2Q_16x128().to(device))
    else:
        raise Exception(f"generator_name error: {generator_name}")

    optimizer, scheduler = build_optimizer_and_scheduler(model, configs=configs)
    if generator is not None:
        g_optimizer, g_scheduler = build_optimizer_and_scheduler(generator, configs=configs)

    if adapter_name == 'vqc':
        model_adapter = VQCmpdAdapter(model=model, train_loader=train_loader, optimizer=optimizer,
                                     scheduler=scheduler,
                                     generator=generator, g_optimizer=g_optimizer, g_scheduler=g_scheduler,
                                     batch_size=configs.batch_size, writer=writer, total_epoch=configs.epochs,
                                     highlight_lambda=configs.highlight_lambda, clip_norm=configs.clip_norm,
                                     device=device, verbose=configs.verbose, m=configs.m, loss_type=configs.loss_type,
                                     advSampleLoader=None, train_strategy=configs.train_strategy)
    elif adapter_name == 'base':
        model_adapter = BaseModelAdapter(model=model, train_loader=train_loader, optimizer=optimizer,
                                     scheduler=scheduler,
                                     generator=None, g_optimizer=None, g_scheduler=None,
                                     batch_size=configs.batch_size, writer=writer, total_epoch=configs.epochs,
                                     highlight_lambda=configs.highlight_lambda, clip_norm=configs.clip_norm,
                                     device=device, verbose=configs.verbose, m=configs.m, loss_type=configs.loss_type,
                                     train_strategy=configs.train_strategy)
    else:
        raise Exception(f"adapter_name error: {adapter_name}")
    if type(generator) is list:
        model_adapter.compound_gen = True
    return model, optimizer, scheduler, model_adapter


def build_optimizer_and_scheduler(model, configs):
    if type(model) is list:
        return build_multi_optimizer_and_scheduler(model, configs)

    no_decay = ['bias', 'layer_norm', 'LayerNorm']  # no decay for parameters of layer norm and bias
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=configs.init_lr)

    if configs.schedule == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, configs.num_train_steps * configs.warmup_proportion,
                                                    configs.num_train_steps)
    elif configs.schedule == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, configs.num_train_steps * configs.warmup_proportion,
                                                    configs.num_train_steps)
    else:
        raise Exception("error")
    return optimizer, scheduler

def build_multi_optimizer_and_scheduler(models, configs):
    optimizers, schedulers = [], []
    for model in models:
        no_decay = ['bias', 'layer_norm', 'LayerNorm']  # no decay for parameters of layer norm and bias
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=configs.init_lr)

        if configs.schedule == "linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, configs.num_train_steps * configs.warmup_proportion,
                                                        configs.num_train_steps)
        elif configs.schedule == "cosine":
            scheduler = get_cosine_schedule_with_warmup(optimizer, configs.num_train_steps * configs.warmup_proportion,
                                                        configs.num_train_steps)
        else:
            raise Exception("error")
        optimizers.append(optimizer)
        schedulers.append(scheduler)
    return optimizers, schedulers
