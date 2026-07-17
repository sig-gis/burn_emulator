from burn_emulator.utils import dynamic_import, read_blob


class BaseRunner
    def __init__(self, model, dataset, dataloader, **kwargs):
        self.model = dynamic_import(model)
        self.dataset = dynamic_import(dataset)
        self.dataloader = dyanamic_import(dataloader, {"dataset": dataset})

        # TODO: dynamically move model to device or instantiate 