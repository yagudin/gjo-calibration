import numpy as np
import plotly.graph_objects as go

from calibration import calibration_curve


def plotly_calibration(y_true, y_pred, n_bins, strategy="quantile"):
    fraction_of_positives, mean_predicted_value, counts = calibration_curve(
        y_true, y_pred, n_bins=n_bins, strategy=strategy
    )
    error_y = np.sqrt((fraction_of_positives) * (1 - fraction_of_positives) / counts)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=mean_predicted_value,
            y=fraction_of_positives,
            customdata=counts,
            mode="markers",
            error_y=dict(
                type="data",
                array=error_y,
                thickness=1.5,
                width=3,
            ),
            hovertemplate="<br>".join(
                [
                    "x: %{x:.3f}",
                    "y: %{y:.3f}",
                    "N: %{customdata}",
                    "<extra></extra>",
                ]
            ),
            showlegend=False,
        )
    )

    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(
            color="LightSeaGreen",
            width=2,
            dash="dot",
        ),
        opacity=0.5,
    )

    fig.update_layout(
        width=800,
        height=800,
        title="Calibration plot",
        xaxis_title="Mean predicted value",
        yaxis_title="Fraction of positives (± std)",
    )

    fig.update_xaxes(
        range=[-0.05, 1.05],
        constrain="domain",
    )

    fig.update_yaxes(
        range=[-0.05, 1.05],
        constrain="domain",
        scaleanchor="x",
        scaleratio=1,
    )

    return fig


def plotly_calibration_odds(y_true, y_pred, n_bins, strategy="quantile"):
    y_pred = np.clip(y_pred, 0.005, 0.995)  # clipping to avoid undefined odds
    y_true = np.clip(y_true, 1e-3, 1 - 1e-3)
    fraction_of_positives, mean_predicted_value, counts = calibration_curve(
        y_true, y_pred, n_bins=n_bins, strategy=strategy
    )
    error_y = np.sqrt((fraction_of_positives) * (1 - fraction_of_positives) / counts)

    fig = go.Figure()

    transform = lambda x: np.log2(1 / (1 - x) - 1)  # 66.6% → 2^{1}:1 → 1

    customdata = np.dstack(
        [
            counts,
            [
                f"{2**x:.1f} : 1" if x > 0 else f"1 : {2**-x:.1f}"
                for x in transform(mean_predicted_value)
            ],
            [
                f"{2**x:.1f} : 1" if x > 0 else f"1 : {2**-x:.1f}"
                for x in transform(fraction_of_positives)
            ],
        ]
    ).squeeze()

    fig.add_trace(
        go.Scatter(
            x=transform(mean_predicted_value),
            y=transform(fraction_of_positives),
            customdata=customdata,
            mode="markers",
            error_y=dict(
                type="data",
                symmetric=False,
                array=transform(fraction_of_positives + error_y)
                - transform(fraction_of_positives),
                arrayminus=transform(fraction_of_positives)
                - transform(fraction_of_positives - error_y),
                thickness=1.5,
                width=3,
            ),
            hovertemplate="<br>".join(
                [
                    "x: %{customdata[1]}",
                    "y: %{customdata[2]}",
                    "N: %{customdata[0]}",
                    "<extra></extra>",
                ]
            ),
            showlegend=False,
        )
    )

    fig.add_shape(
        type="line",
        x0=-8,
        y0=-8,
        x1=8,
        y1=8,
        line=dict(
            color="LightSeaGreen",
            width=2,
            dash="dot",
        ),
        opacity=0.5,
    )

    fig.update_layout(
        width=800,
        height=800,
        title="Calibration plot in terms of odds",
        xaxis_title="Mean predicted value",
        yaxis_title="Fraction of positives (± std)",
    )

    fig.update_xaxes(
        range=[-8, 8],
        constrain="domain",
        tickmode="array",
        tickvals=list(range(-10, 10)),
        ticktext=[
            f"{2**x} : 1" if x > 0 else f"1 : {2**-x}" for x in list(range(-10, 10))
        ],
    )

    fig.update_yaxes(
        range=[-8, 8],
        constrain="domain",
        scaleanchor="x",
        scaleratio=1,
        tickvals=list(range(-10, 10)),
        ticktext=[
            f"{2**x} : 1" if x > 0 else f"1 : {2**-x}" for x in list(range(-10, 10))
        ],
    )

    return fig
