import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import comet_ml


@dataclass
class Metrics:
    sdf_loss: float
    a_star_loss: float
    optimality_loss: float
    physics_loss: float
    t_loss: float
    warming: float
    boundary_loss: float
    step: int


@dataclass
class LossLogging:
    a_star_min_points: list[int] = field(default_factory=list)
    x_derivative: np.ndarray = field(default_factory=lambda: np.zeros(1))
    y_derivative: np.ndarray = field(default_factory=lambda: np.zeros(1))
    v_derivative: np.ndarray = field(default_factory=lambda: np.zeros(1))
    theta_derivative: np.ndarray = field(default_factory=lambda: np.zeros(1))


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
                "loss/boundary": metrics.boundary_loss,
                "loss/T": metrics.t_loss,
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
        plot_logger: LossLogging,
        t_steps,
    ) -> None:
        print(f"loss: {loss:>7f}")
        path_x = output[:, 0]
        path_y = output[:, 1]
        v_list = output[:, 2]
        theta_list = output[:, 3]
        x_dot = plot_logger.x_derivative
        y_dot = plot_logger.y_derivative
        v_dot_calc = np.sqrt(x_dot * x_dot + y_dot * y_dot)
        v_dot = plot_logger.v_derivative
        theta_dot = plot_logger.theta_derivative
        a_list = output[:, 4]
        omega_list = output[:, 5]
        a_star_point = plot_logger.a_star_min_points
        fig2, ax3 = plt.subplots()
        fig, (ax1, ax2) = plt.subplots(2, 2)
        ax3.plot(path_x, path_y, color="orange", label="Path")
        # ax1[0].scatter(path_x, path_y)
        ax1[0].plot(t_steps, v_list, label=r"$v$")
        ax1[0].plot(
            t_steps, v_dot_calc, label=r"$\sqrt{\dot{x}^2 + \dot{y}^2}$", linestyle="--"
        )
        ax2[0].plot(t_steps, theta_list, label=r"$\theta$")
        ax1[1].plot(t_steps, a_list, label=r"$a$")
        ax1[1].plot(t_steps, v_dot, label=r"$\dot{v}$", linestyle="--")
        ax2[1].plot(t_steps, omega_list, label=r"$\omega$")
        ax2[1].plot(t_steps, theta_dot, label=r"$\dot{\theta}$", linestyle="--")
        ax3.imshow(sdf, origin="lower", cmap="Greys")
        if turning_points.size > 0:
            ax3.scatter(
                turning_points[:, 0],
                turning_points[:, 1],
                color="magenta",
                marker="*",
                label="Guiding points",
            )
        sp = start_pos
        ax3.scatter(sp[0], sp[1], label="Start point", color="limegreen", marker="o")
        ep = end_pos
        ax3.scatter(ep[0], ep[1], label="End point", color="red", marker="x")
        # Plot the point the A-star loss is based on, i.e. the closest point on the path
        # ax1[0].scatter(
        #    path_x[a_star_point],
        #    path_y[a_star_point],
        #    color="yellow",
        #    marker="1",
        # )
        ax2[0].legend()
        ax3.legend()
        ax1[1].legend()
        ax1[0].legend()
        ax2[1].legend()
        self.experiment_.log_figure(figure_name="Map", figure=fig2, step=step)
        self.experiment_.log_figure(figure_name="State Values", figure=fig, step=step)
        plt.close(fig)
        plt.close(fig2)

    def end_experiment(self) -> None:
        self.experiment_.end()
