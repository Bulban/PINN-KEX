import comet_ml
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class Metrics:
    sdf_loss: float
    a_star_loss: float
    optimality_loss: float
    physics_loss: float
    warming: float
    step: int


class Logger:
    def __init__(self, experiment: comet_ml.CometExperiment) -> None:
        self.experiment_ = experiment

    def log_metrics(self, metrics: Metrics) -> None:
        self.experiment_.log_metrics(
            {
                "loss/sdf": metrics.sdf_loss,
                "loss/a_star": metrics.a_star_loss,
                "loss/physics": metrics.physics_loss,
                "loss/optimality": metrics.optimality_loss,
                "loss/warming": metrics.warming,
            },
            step=metrics.step,
        )

    def get_experiment(self) -> comet_ml.CometExperiment:
        return self.experiment_

    def log_parameters(self, hyper_params) -> None:
        self.experiment_.log_parameters(hyper_params)

    def log_figure(
        self,
        loss: float,
        output,
        sdf,
        start_pos,
        end_pos,
        turning_points,
        step,
        a_star_point,
    ) -> None:
        print(f"loss: {loss:>7f}")
        path_x = []
        path_y = []
        v_list = []
        theta_list = []
        a_list = []
        omega_list = []
        path_x = output[:, 0]
        path_y = output[:, 1]
        v_list = output[:, 2]
        theta_list = output[:, 3]
        a_list = output[:, 4]
        omega_list = output[:, 5]
        fig, (ax1, ax2) = plt.subplots(2, 2)
        ax1[0].plot(path_x, path_y, color="orange")
        ax1[0].scatter(path_x, path_y)
        ax2[0].plot(v_list, label=r"$v$")
        ax2[0].plot(theta_list, label=r"$\theta$")
        ax1[1].plot(a_list, label=r"$a$")
        ax2[1].plot(omega_list, label=r"$\omega$")
        ax1[0].imshow(sdf, origin="lower")
        ax1[0].scatter(
            turning_points[:, 0], turning_points[:, 1], color="magenta", marker="*"
        )
        sp = start_pos
        ax1[0].scatter(sp[0], sp[1], color="limegreen", marker="o")
        ep = end_pos
        ax1[0].scatter(ep[0], ep[1], color="red", marker="x")
        # Plot the point the A-star loss is based on, i.e. the closest point on the path
        ax1[0].scatter(
            path_x[a_star_point],
            path_y[a_star_point],
            color="yellow",
            marker="1",
        )
        ax2[0].legend()
        ax1[1].legend()
        ax2[1].legend()
        self.experiment_.log_figure(fig, step=step)
        plt.close()

    def end_experiment(self) -> None:
        self.experiment_.end()
