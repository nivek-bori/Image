import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

### Detection, Prediction, Tracking

def annotate_detections(annotated_frame, *boxes):
    # curr_det: [cx, cy, w, h]

    for detections, color in boxes:
        for det in detections:
            xywh = det.xywh_v

            # detection pos
            x1 = int(xywh[0] - xywh[2] / 2)
            x2 = int(xywh[0] + xywh[2] / 2)
            y1 = int(xywh[1] - xywh[3] / 2)
            y2 = int(xywh[1] + xywh[3] / 2)

            # detection bounding box
            cv2.circle(
                annotated_frame, (int(xywh[0]), int(xywh[1])), 6, color, 2
            )  # raw xy
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # detection text
            text = f'conf: {det.conf[0]:.4g}'
            cv2.putText(
                annotated_frame,
                text,
                (int(x1), int(y2 + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

    return annotated_frame


def annotate_tracklets(annotated_frame, tracks, color):
    for id, track in tracks.items():
        bbox = track.k_filter.x[:, 0]  # x: [cx, cy, width, height, vx, vy, vw, vh]

        # tracking pos
        w, h = bbox[2], bbox[3]
        x1 = int(bbox[0] - w / 2)
        y1 = int(bbox[1] - h / 2)
        x2 = int(bbox[0] + w / 2)
        y2 = int(bbox[1] + h / 2)

        # tracking bounding box
        cv2.circle(
            annotated_frame, (int(bbox[0]), int(bbox[1])), 6, color, 2
        )  # raw cxy
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        # tracking text
        text = f'id: {id}'
        cv2.putText(
            annotated_frame,
            text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

    return annotated_frame


def annotate_predictions(annotated_frame, tracks, color):
    for track in tracks.values():
        bbox = track.k_filter.x[:, 0]  # x: [cx, cy, width, height, vx, vy, vw, vh]

        # prediction pos
        pred_w, pred_h = bbox[2] + bbox[6], bbox[3] + bbox[7]
        pred_x1 = int((bbox[0] + bbox[4]) - pred_w / 2)
        pred_y1 = int((bbox[1] + bbox[5]) - pred_h / 2)
        pred_x2 = int((bbox[0] + bbox[4]) + pred_w / 2)
        pred_y2 = int((bbox[1] + bbox[5]) + pred_h / 2)

        # prediction bounding box
        cv2.circle(
            annotated_frame, (int(bbox[0]), int(bbox[1])), 6, color, 2
        )  # raw cxy
        cv2.rectangle(annotated_frame, (pred_x1, pred_y1), (pred_x2, pred_y2), color, 2)

    return annotated_frame


def annotate_n_predictions(annotated_frame, pred_tracks, color):
    for track in pred_tracks:
        for i, bbox in enumerate(
            track[1:]
        ):  # do not include first bbox (non-prediction bbox)
            # prediction pos
            w, h = bbox[2, 0], bbox[3, 0]
            x1 = int(bbox[0, 0] - w / 2)
            y1 = int(bbox[1, 0] - h / 2)
            x2 = int(bbox[0, 0] + w / 2)
            y2 = int(bbox[1, 0] + h / 2)

            # prediction bounding box
            weight = 1 - 0.8 * (i / len(track))
            weighted_color = [weight * c for c in color]
            cv2.circle(
                annotated_frame,
                (int(bbox[0, 0]), int(bbox[1, 0])),
                int(weight * 6),
                weighted_color,
                2,
            )  # raw cxy
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), weighted_color, 2)

    return annotated_frame


### Logging/Visualzing

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_average_metrics(eval_summaries, title='Average Tracking Performance'):
    if not eval_summaries:
        return
    
    all_summaries_df = pd.DataFrame(eval_summaries).T
    metrics_to_plot = ['mota', 'motp', 'idf1']
    avg_metrics = all_summaries_df[metrics_to_plot].mean()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(avg_metrics.index, avg_metrics.values, color=['#2E86AB', '#A23B72', '#F18F01'], )
    
    ax.set_title(title)
    ax.set_ylabel('Score')
    ax.set_xlabel('Metric')
    
    # Add value labels on bars
    for bar, value in zip(bars, avg_metrics.values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() / 2, f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(avg_metrics.values) * 1.1)
    
    plt.show()


def plot_camera_comparison(eval_summaries, metric_name='mota', title_prefix='Tracking Performance'):
    if not eval_summaries:
        return

    camera_names = list(eval_summaries.keys())
    metric_values = [summary[metric_name] for summary in eval_summaries.values()]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(camera_names, metric_values, color='#4A90E2', alpha=0.8)
    
    ax.set_title(f'{title_prefix}: {metric_name.upper()} by Camera')
    ax.set_ylabel(metric_name.upper())
    ax.set_xlabel('Camera')
    
    # Add value labels
    for bar, value in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() / 2.0, f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.grid(axis='x', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.show()


def plot_error_distribution(eval_summaries, error_metrics=['idfp', 'idfn', 'idsw'], 
                           title='Average Error Distribution'):
    """Simple pie chart for error distribution"""
    if not eval_summaries:
        return

    all_summaries_df = pd.DataFrame(eval_summaries).T
    
    # Handle missing columns gracefully
    available_metrics = [m for m in error_metrics if m in all_summaries_df.columns]
    if not available_metrics:
        print(f"Warning: None of the requested metrics {error_metrics} found in data")
        return
    
    total_errors = all_summaries_df[available_metrics].sum()
    
    if total_errors.sum() == 0:
        print("No errors found in the data")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][:len(available_metrics)]
    
    wedges, texts, autotexts = ax.pie(total_errors, 
                                     labels=[m.upper() for m in available_metrics],
                                     autopct='%1.1f%%', startangle=90,
                                     colors=colors)
    
    ax.set_title(title)
    
    plt.show()


def plot_metric_trends(eval_summaries, metric_name='mota', title_prefix='Metric Trend'):
    """Simple line plot for metric trends"""
    if not eval_summaries:
        return

    camera_names = list(eval_summaries.keys())
    metric_values = [summary[metric_name] for summary in eval_summaries.values()]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(camera_names, metric_values, marker='o', linestyle='-', color='#4A90E2', linewidth=2, markersize=6)
    
    ax.set_title(f'{title_prefix}: {metric_name.upper()} Across Cameras')
    ax.set_ylabel(metric_name.upper())
    ax.set_xlabel('Camera')
    
    ax.grid(True, alpha=0.3)
    plt.show()


def plot_multi_metric_comparison(eval_summaries, metrics=['mota', 'motp', 'idf1'], title='Multi-Metric Comparison'):
    if not eval_summaries:
        return
    
    all_summaries_df = pd.DataFrame(eval_summaries).T
    available_metrics = [m for m in metrics if m in all_summaries_df.columns]
    
    if not available_metrics:
        print(f"Warning: None of the requested metrics {metrics} found in data")
        return
    
    camera_names = list(eval_summaries.keys())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(camera_names))
    width = 0.25
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for i, metric in enumerate(available_metrics):
        values = [summary[metric] for summary in eval_summaries.values()]
        ax.bar(x + i * width, values, width, label=metric.upper(), color=colors[i % len(colors)], alpha=0.8)
    
    ax.set_title(title)
    ax.set_ylabel('Score')
    ax.set_xlabel('Camera')
    ax.set_xticks(x + width)
    ax.set_xticklabels(camera_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='x', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)

    plt.show()


def plot_performance_summary(eval_summaries, title='Performance Summary'):
    """Simple 2x2 subplot layout with key metrics"""
    if not eval_summaries:
        return
    
    all_summaries_df = pd.DataFrame(eval_summaries).T
    camera_names = list(eval_summaries.keys())
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Average metrics
    metrics_to_plot = ['mota', 'motp', 'idf1']
    avg_metrics = all_summaries_df[metrics_to_plot].mean()
    ax1.bar(avg_metrics.index, avg_metrics.values, 
            color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.8)
    ax1.set_title('Average Performance', fontweight='bold')
    ax1.set_ylabel('Score')
    ax1.grid(axis='x', alpha=0.5)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: MOTA by camera
    mota_values = [summary['mota'] for summary in eval_summaries.values()]
    ax2.bar(camera_names, mota_values, color='#4A90E2', alpha=0.8)
    ax2.set_title('MOTA by Camera', fontweight='bold')
    ax2.set_ylabel('MOTA')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='x', alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Error distribution (if available)
    error_metrics = ['idfp', 'idfn']
    available_error_metrics = [m for m in error_metrics if m in all_summaries_df.columns]
    if available_error_metrics:
        total_errors = all_summaries_df[available_error_metrics].sum()
        ax3.pie(total_errors, labels=[m.upper() for m in available_error_metrics],
                autopct='%1.1f%%', colors=['#FF6B6B', '#4ECDC4'])
        ax3.set_title('Error Distribution', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No error data\navailable', ha='center', va='center',
                transform=ax3.transAxes)
        ax3.set_title('Error Distribution', fontweight='bold')
    
    # Plot 4: MOTA trend
    ax4.plot(camera_names, mota_values, marker='o', color='#4A90E2', linewidth=2)
    ax4.set_title('MOTA Trend', fontweight='bold')
    ax4.set_ylabel('MOTA')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.show()