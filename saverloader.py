import torch
import os, pathlib
# import hyperparams as hyp
import numpy as np

def save_ensemble(ckpt_dir, optimizer, models, models_ema, global_step, keep_latest=10):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    prev_ckpts = list(pathlib.Path(ckpt_dir).glob('model-*'))
    prev_ckpts.sort(key=lambda p: p.stat().st_mtime,reverse=True)
    if len(prev_ckpts) > keep_latest-1:
        for f in prev_ckpts[keep_latest-1:]:
            f.unlink()
    model_path = '%s/model-%09d.pth' % (ckpt_dir, global_step)
    
    ckpt = {'optimizer_state_dict': optimizer.state_dict()}
    for i in range(len(models)):
        ckpt['model_state_dict_{}'.format(i)] = models[i].state_dict()
        ckpt['ema_model_state_dict_{}'.format(i)] = models_ema[i].state_dict()
    torch.save(ckpt, model_path)
    print("saved a checkpoint: %s" % (model_path))

def save(ckpt_dir, optimizer, model, global_step, scheduler=None, model_ema=None, keep_latest=5, model_name='model'):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    prev_ckpts = list(pathlib.Path(ckpt_dir).glob('%s-*' % model_name))
    prev_ckpts.sort(key=lambda p: p.stat().st_mtime,reverse=True)
    if len(prev_ckpts) > keep_latest-1:
        for f in prev_ckpts[keep_latest-1:]:
            f.unlink()
    model_path = '%s/%s-%09d.pth' % (ckpt_dir, model_name, global_step)
    
    ckpt = {'optimizer_state_dict': optimizer.state_dict()}
    ckpt['model_state_dict'] = model.state_dict()
    if scheduler is not None:
        ckpt['scheduler_state_dict'] = scheduler.state_dict()
    if model_ema is not None:
        ckpt['ema_model_state_dict'] = model_ema.state_dict()
    torch.save(ckpt, model_path)
    print("saved a checkpoint: %s" % (model_path))

def load_ensemble(ckpt_dir, optimizer, models, models_ema):
    print('reading ckpt from %s' % ckpt_dir)
    checkpoint_dir = os.path.join('checkpoints/', ckpt_dir)
    step = 0
    if not os.path.exists(checkpoint_dir):
        print('...there is no full checkpoint here!')
    else:
        ckpt_names = os.listdir(checkpoint_dir)
        steps = [int((i.split('-')[1]).split('.')[0]) for i in ckpt_names]
        if len(ckpt_names) > 0:
            step = max(steps)
            model_name = 'model-%09d.pth' % (step)
            path = os.path.join(checkpoint_dir, model_name)
            print('...found checkpoint %s'%(path))

            checkpoint = torch.load(path)

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for i, (model, model_ema) in enumerate(zip(models, models_ema)):
                model.load_state_dict(checkpoint['model_state_dict_{}'.format(i)])
                model_ema.load_state_dict(checkpoint['ema_model_state_dict_{}'.format(i)])
        else:
            print('...there is no full checkpoint here!')
    return step


def load(ckpt_dir, model, optimizer=None, scheduler=None, model_ema=None, step=0, model_name='model'):
    print('reading ckpt from %s' % ckpt_dir)
    if not os.path.exists(ckpt_dir):
        print('...there is no full checkpoint here!')
        print('-- note this function no longer appends "saved_checkpoints/" before the ckpt_dir --')
    else:
        ckpt_names = os.listdir(ckpt_dir)
        steps = [int((i.split('-')[1]).split('.')[0]) for i in ckpt_names]
        if len(ckpt_names) > 0:
            if step==0:
                step = max(steps)
            model_name = '%s-%09d.pth' % (model_name, step)
            path = os.path.join(ckpt_dir, model_name)
            print('...found checkpoint %s'%(path))

            checkpoint = torch.load(path)

            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if model_ema is not None:
                model_ema.load_state_dict(checkpoint['ema_model_state_dict']) 
        else:
            print('...there is no full checkpoint here!')
    return step
