from torch.optim.lr_scheduler import _LRScheduler


class TTUR_GANLR(_LRScheduler):
    def __init__(self, optimizer, start_decay_step, decay_step, generator_type, last_epoch=-1):
        '''

        :param optimizer:
        :param start_decay_step:
        :param decay_step:
        :param generator_type: ['g','d']
        :param last_epoch:
        '''
        self.start_decay = start_decay_step  # 开始下降的epoch
        self.ttur_type = generator_type
        self.decay_step = decay_step  # 下降多少步到0
        super(TTUR_GANLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1
        # lr不变
        if epoch <= self.start_decay:
            return self.base_lrs
        new_lrs = []
        for base_lr in self.base_lrs:
            lrd = base_lr / self.decay_step
            new_lr_base = base_lr - (lrd * (epoch - self.start_decay))
            if type is 'g':
                new_lr_base = new_lr_base / 2
            elif type is 'd':
                new_lr_base = new_lr_base * 2
            new_lrs.append(new_lr_base)
        return new_lrs
