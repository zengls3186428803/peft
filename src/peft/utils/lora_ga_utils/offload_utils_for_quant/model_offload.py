from .saved_tensor_offload import SavedTensorOffloadContext
from .forward_backward_offload import ForwardBackwardOffloadHookContext


class ModelOffloadHookContext:
    def __init__(
            self,
            model,
            no_split_module_classes=None,
            num_block: int = 2,
            enable=True,
            # =========================
            device="cuda",
            strategy="block",
            with_backward_hook=False
    ):
        self.enable = enable
        if not enable:
            return
        self.forwardBackwardOffloadHookContext = ForwardBackwardOffloadHookContext(
            model=model,
            device=device,
            no_split_module_classes=no_split_module_classes,
            with_backward_hook=with_backward_hook,  # for debug
            enable=True,
            num_block=num_block,
            strategy=strategy,  # enum["module","block"],
        )
        self.savedTensorOffloadContext = SavedTensorOffloadContext()

    def __enter__(self):
        if not self.enable:
            return
        self.forwardBackwardOffloadHookContext.__enter__()
        self.savedTensorOffloadContext.__enter__()
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enable:
            return
        self.forwardBackwardOffloadHookContext.__exit__(exc_type, exc_val, exc_tb)
        self.savedTensorOffloadContext.__exit__(exc_type, exc_val, exc_tb)
        pass
