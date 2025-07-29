from tkinter import messagebox, simpledialog, Canvas
from utils import save_example_to_json, load_dataset_from_json
from rede_neural import RedeNeural
from ball import Ball

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Desenhe com 20 bolinhas")

        self.canvas = Canvas(root, width=200, height=250, bg="white")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)

        self.balls = []
        self.rna = RedeNeural()
        self.json_path = "public/dataset.json"

        dataset = load_dataset_from_json(self.json_path)
        if dataset:
            self.rna.treinar(dataset, epocas=1000)

    def on_click(self, event):
        if len(self.balls) >= 20:
            return
        self.balls.append(Ball(event.x, event.y, "black"))
        self.canvas.create_oval(event.x-5, event.y-5, event.x+5, event.y+5, fill="black")
        if len(self.balls) == 20:
            self.classify_and_ask_to_save()

    def get_projection(self):
        width, height = 200, 250
        proj_v = [0] * 10
        proj_h = [0] * 10
        for b in self.balls:
            iv = min(b.y * 10 // height, 9)
            ih = min(b.x * 10 // width, 9)
            proj_v[iv] += 1
            proj_h[ih] += 1
        return proj_v + proj_h

    def classify_and_ask_to_save(self):
        xk = self.get_projection()
        pred = self.rna.prever(xk)
        texto = "🌳 É um boneco palito!" if pred == 1 else "❌ Não é um boneco palito!"
        messagebox.showinfo("Classificação", texto)

        label = simpledialog.askinteger("Salvar exemplo", "Qual o rótulo? (1 ou 0)", minvalue=0, maxvalue=1)
        if label in [0, 1]:
            save_example_to_json(self.json_path, xk, label)
            messagebox.showinfo("Salvo", "Exemplo salvo!")
        else:
            messagebox.showinfo("Ignorado", "Exemplo não salvo.")

        dataset = load_dataset_from_json(self.json_path)
        if dataset:
            self.rna.treinar(dataset, epocas=1000)

        self.reset_canvas()

    def reset_canvas(self):
        self.canvas.delete("all")
        self.balls = []