import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

import pandas as pd
from shapely.geometry import box, Point
from shapely.ops import unary_union

import imageio
import io
from IPython.display import Image, display

def coverage_area_shapely(df: pd.DataFrame,
                          xmin=0, ymin=0, xmax=10, ymax=10,
                          x_col="center_x_coordinate",
                          y_col="center_y_coordinate",
                          r_col="tower_connection_radius",
                          resolution: int = 64) -> float:
    square = box(xmin, ymin, xmax, ymax)

    clipped_shapes = []
    for x, y, r in df[[x_col, y_col, r_col]].itertuples(index=False, name=None):
        circle = Point(float(x), float(y)).buffer(float(r), resolution=resolution)  # більше resolution -> точніше коло
        clipped_shapes.append(circle.intersection(square))                 # обрізка квадратом

    union = unary_union(clipped_shapes)  # об'єднання покриття
    return float(union.area)

class StantionsVisualizer:
    REQUIRED_COLS = {
        "tower_id",
        "tower_connection_radius",
        "center_x_coordinate",
        "center_y_coordinate",
    }

    def __init__(
        self,
        init_stantions_data: pd.DataFrame,
        space_size: tuple[int, int],
        border_coef: float = 0.1,
        resolution: int = 250
    ):
        missing = self.REQUIRED_COLS - set(init_stantions_data.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        self.station_state_df = init_stantions_data.copy()
        self.space_size = space_size
        self.border_coef = border_coef
        self.resolution = resolution

    def visualize_stantions(self, stantions_data: pd.DataFrame | None = None, show: bool = True):
        if stantions_data is None:
            stantions_data = self.station_state_df

        w, h = self.space_size
        bx, by = w * self.border_coef, h * self.border_coef

        fig, ax = plt.subplots(figsize=(6, 6))

        # space border (0..w, 0..h)
        ax.add_patch(
            Rectangle(
                (0, 0), w, h,
                fill=False,
                edgecolor="red",
                linestyle="--",
                linewidth=1.5,
            )
        )

        for _, row in stantions_data.iterrows():
            x = float(row["center_x_coordinate"])
            y = float(row["center_y_coordinate"])
            r = float(row["tower_connection_radius"])
            tid = row["tower_id"]

            ax.add_patch(Circle((x, y), r, facecolor="#1f77b4", edgecolor="none", alpha=0.35))
            ax.text(x, y, str(tid), ha="center", va="center", fontsize=9, color="black")

        ax.set_xlim(-bx, w + bx)
        ax.set_ylim(-by, h + by)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        area = coverage_area_shapely(stantions_data, resolution=self.resolution)
        not_covered = (w*h - area) / (w*h) * 100
        ax.set_title(f"Not Covered ~ {not_covered:.1f}%")
        
        if show:
            print(show)
            plt.show()

        return fig, ax
    
    def visualize_coverage_with_stations(self, clean_heatmap, stantions_data: pd.DataFrame | None = None):
            """
            Відображає хітмап ймовірності покриття разом із колами станцій.
            clean_probability: тензор розміром (2500,) або (50, 50)
            """
            if stantions_data is None:
                stantions_data = self.station_state_df

            w, h = self.space_size
            bx, by = w * self.border_coef, h * self.border_coef
            
            # Інвертуємо, щоб візуалізувати ПОКРИТТЯ (coverage = 1 - clean)
            # Так яскраві зони будуть там, де є зв'язок
            coverage_heatmap = 1 - clean_heatmap

            fig, ax = plt.subplots(figsize=(8, 8))

            # 2. Малюємо хітмап
            # extent=[xmin, xmax, ymin, ymax] синхронізує індекси масиву з координатами
            im = ax.imshow(
                coverage_heatmap, 
                extent=[0, w, 0, h], 
                origin='lower', 
                cmap='viridis', 
                alpha=0.6  # Напівпрозорість, щоб бачити сітку та кола
            )
            
            # Додаємо колірну шкалу
            fig.colorbar(im, ax=ax, label='Ймовірність покриття (max(p))')

            # 3. Малюємо межі простору (червоний пунктир)
            ax.add_patch(
                Rectangle(
                    (0, 0), w, h,
                    fill=False,
                    edgecolor="red",
                    linestyle="--",
                    linewidth=2,
                    label="Space Border"
                )
            )

            # 4. Малюємо станції (кола та ID)
            for _, row in stantions_data.iterrows():
                x = float(row["center_x_coordinate"])
                y = float(row["center_y_coordinate"])
                r = float(row["tower_connection_radius"])
                tid = row["tower_id"]

                # Малюємо лише контур кола, щоб бачити, як він збігається з хітмапом
                ax.add_patch(
                    Circle((x, y), r, facecolor="none", edgecolor="#1f77b4", linewidth=1.5, alpha=0.8)
                )
                ax.text(x, y, str(tid), ha="center", va="center", fontsize=10, 
                        color="white", weight='bold', bbox=dict(facecolor='black', alpha=0.2, lw=0))

            # Налаштування осей
            ax.set_xlim(-bx, w + bx)
            ax.set_ylim(-by, h + by)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.2)
            ax.set_xlabel("X coordinate")
            ax.set_ylabel("Y coordinate")
            ax.set_title("Overlay: Coverage Probability Heatmap & Station Radii")

            plt.tight_layout()
            plt.show()
            return fig, ax


def save_as_gif(vis_figures, filename='task_1_coverage_evolution.gif', fps=5):
    """
    Перетворює збережені фігури на GIF анімацію.
    fps: Frames Per Second (швидкість анімації). Більше = швидше.
    """
    if not vis_figures:
        print("Немає збережених фігур для анімації.")
        return

    print(f"Збираємо {len(vis_figures)} кадрів у GIF...")
    images = []

    # Ітеруємося по збережених даних [fig, ax, i, current_lr]
    for item in vis_figures:
        fig = item[0] 
        step_num = item[2]
        
        # Додаємо заголовок до поточної фігури перед збереженням
        # (Якщо його там ще немає або треба оновити)
        ax = item[1]
        original_title = ax.get_title()
        ax.set_title(f"Step: {step_num} | {original_title}")

        # --- Магія збереження в пам'ять ---
        # Створюємо буфер у пам'яті
        buf = io.BytesIO()
        # Зберігаємо фігуру в цей буфер у форматі PNG
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        # Перемотуємо буфер на початок
        buf.seek(0)
        # Читаємо зображення з буфера за допомогою imageio
        images.append(imageio.v2.imread(buf))
        # Закриваємо буфер
        buf.close()
        
    # Зберігаємо список зображень як GIF
    imageio.mimsave(filename, images, fps=fps, loop=0) # loop=0 означає нескінченний повтор
    print(f"GIF успішно збережено як {filename}")
    
    # Відображаємо GIF у ноутбуці
    display(Image(data=open(filename,'rb').read(), format='png'))
