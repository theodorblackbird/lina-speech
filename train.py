from pytorch_lightning.cli import LightningCLI

from model.lina import cli

if __name__ == "__main__":
    # import socket
    # if socket.gethostname().split(".")[0] in ["gottan", "guqin", "gusli"]:
    #    import manage_gpus as gpl
    #    gpl.get_gpu_lock()
    cli()
