# BackPropagation_WebApp

An interactive web-based application built using React that visualizes the **Backpropagation algorithm** in artificial neural networks. This tool is designed for educational and research purposes to help users intuitively understand the inner mechanics of feedforward propagation and gradient-based learning.

---

## 📌 Project Objectives

- **Visualize forward and backward passes** in a multi-layer neural network
- **Demonstrate weight adjustments** using gradient descent
- Provide an **interactive UI** for defining inputs, weights, and observing the learning process in real time

---

## 🧠 Core Concepts

This web app simulates:

- **Feedforward Propagation:** Computing outputs from user-defined inputs through activation functions.
- **Backpropagation Algorithm:** Adjusting weights using calculated loss gradients.
- **Error Minimization:** Updating weights to minimize prediction error using the derivative chain rule.

The mathematical engine is implemented manually — **without using ML libraries** — to reinforce a conceptual understanding of how each layer computes its outputs and gradients.

---

## ⚙️ Technologies Used

| Component        | Technology        |
|------------------|------------------|
| Frontend         | React.js (CRA)   |
| UI Components    | HTML5, CSS3      |
| Logic Engine     | JavaScript (Vanilla Math) |
| State Management | React State Hooks |
| Version Control  | Git & GitHub     |

---

## 📁 Project Structure

```plaintext
backprop-web-app/
│
├── src/
│   ├── App.js                     # Root component
│   ├── BackpropagationApp.js     # Main visualizer logic and UI container
│   ├── NeuralNetworkVisualizer.js# Canvas drawing for network structure
│   ├── BackpropMathEngine.js     # Custom math engine for feedforward & backprop
│   ├── index.js                  # React entry point
│   └── ...                       # CRA boilerplate files
│
├── public/
│   └── index.html                # App container
│
├── package.json
└── README.md                     # You're here
````

---

## 🚀 Getting Started

### 🔧 Prerequisites

* [Node.js](https://nodejs.org/) (v14+)
* npm (comes with Node.js)

### 📦 Installation & Run

```bash
git clone https://github.com/ShivangiChouhan/BackPropagation_WebApp.git
cd BackPropagation_WebApp
npm install
npm start
```

Then open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 🧪 Educational Use Case

This tool can be used for:

* **BTech/BE academic demonstrations**
* **AI/ML course tutorials**
* **Visual-based concept reinforcement**
* **Quick testing of backpropagation effects without external libraries**

---

## 🧮 Algorithm Summary

The weight update rule follows:

```
W := W - η * ∇E
```

Where:

* `W` = weight
* `η` = learning rate
* `∇E` = gradient of error with respect to weight

The gradients are computed using the **chain rule**, and the activation functions (e.g., sigmoid) are manually implemented in `BackpropMathEngine.js`.

---

## 📌 Author

**Shivangi Chouhan**
Electronics and Telecommunication Engineering
Thakur College of Engineering and Technology, Mumbai
🔗 [GitHub Profile](https://github.com/ShivangiChouhan)

---

## 📜 License

This project is open-source 

