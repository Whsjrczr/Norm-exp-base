import torch
import torch.nn as nn
import wandb

class network_hook():
    def __init__(self, model, targets):
        super(network_hook, self).__init__()
        self.hook_handles = []
        self.targets = targets
        self.layer_info = {
            'inputs': {},
            'outputs': {},
            'gradients': {}
        }
        self.register_hook(model)
        # self.last_log = {}

    
    def register_hook(self, model):
        for name, module in model.named_modules():
            if any(isinstance(module,  target_type) for target_type in self.targets):
            # if module in self.targets:
                forward_hook, backward_hook = self.get_layer_info(name)
                handle_forward = module.register_forward_hook(forward_hook)
                handle_backward = module.register_backward_hook(backward_hook)
                self.hook_handles.extend([handle_forward, handle_backward])

    def get_layer_info(self, name):
        def forward_hook(module, input, output):
            self.layer_info['inputs'][name] = input[0].detach()
            self.layer_info['outputs'][name] = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.layer_info['gradients'][name] = grad_output[0].detach()

        return forward_hook, backward_hook

    def log(self):
        log_dict = {}
        for name in self.layer_info['inputs']:
            input_log = f"{name}_input"
            output_log = f"{name}_output"
            gradient_log = f"{name}_gradient"
            
            if name in self.layer_info['inputs']:
                log_dict[input_log] = self.layer_info['inputs'][name].mean().item()
            if name in self.layer_info['outputs']:
                log_dict[output_log] = self.layer_info['outputs'][name].mean().item()
            if name in self.layer_info['gradients']:
                log_dict[gradient_log] = self.layer_info['gradients'][name].mean().item()

        # self.last_log = log_dict
            
        wandb.log(log_dict)
        self.clean_hook()

    def clean_hook(self):
        self.layer_info['inputs'].clear()
        self.layer_info['outputs'].clear()
        self.layer_info['gradients'].clear()
    
    def remove_hook(self):
        for handle in self.hook_handles:
            handle.remove()
