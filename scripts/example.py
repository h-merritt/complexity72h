import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    inner: list[str] = list(cfg.components.inner)
    outer: list[str] = list(cfg.components.outer)
    print(f"inner: {inner}")
    print(f"outer: {outer}")


if __name__ == "__main__":
    main()
