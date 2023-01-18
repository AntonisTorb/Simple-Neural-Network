import sys
import os
import numpy as np
import PySimpleGUI as sg


# Following 2 lines needed to import the neural network from 2 directories higher, as relative imports lead to ImportError
# This way I don't need to maintain a separate snn.py file
snn_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(snn_path)

from snn import SimpleNeuralNetwork

def main() -> None:
    '''Main function.'''
    
    hidden_nodes = 3
    hidden_layers = 1
    learning_rate = 0.05

    layout = [
        [sg.Push(), sg.Text("CPU:"), sg.Input(0, (4,1), readonly=True, justification="center", key="-CPU_SCORE-"), sg.Push()],
        [sg.Text("CPU Choice:", size=(10,1)), sg.Push(), 
            sg.Button("Rock", key="-CPU_ROCK-", size=(8,1)), sg.Push(), 
            sg.Button("Paper", key="-CPU_PAPER-", size=(8,1)), sg.Push(), 
            sg.Button("Scissors", key="-CPU_SCISSORS-", size=(8,1)), sg.Push()],
        [sg.Text("CPU Odds:", size=(10,1)), sg.Push(), 
            sg.Input(justification="center", readonly=True, key="-CPU_ROCK_CALC-", size=(8,1)), sg.Push(), 
            sg.Input(justification="center", readonly=True, key="-CPU_PAPER_CALC-", size=(8,1)), sg.Push(), 
            sg.Input(justification="center", readonly=True, key="-CPU_SCISSORS_CALC-", size=(8,1)), sg.Push()],
        [sg.HorizontalSeparator()],
        [sg.Push(), sg.Text("Ties:"), sg.Input(0, (4,1), readonly=True, justification="center", key="-TIES-"), sg.Push()],
        [sg.HorizontalSeparator()],
        [sg.Text("User Choice:", size=(10,1)), sg.Push(), sg.Button("Rock", key="-USER_ROCK-", size=(8,1)), sg.Push(), 
            sg.Button("Paper", key="-USER_PAPER-", size=(8,1)), sg.Push(), 
            sg.Button("Scissors", key="-USER_SCISSORS-", size=(8,1)), sg.Push()],
        [sg.Push(), sg.Text("You:"), sg.Input(0, (4,1), readonly= True, justification="center", key = "-USER_SCORE-"), sg.Push()],
        [sg.HorizontalSeparator()],
        [sg.Button("Reset"), sg.Push(), sg.Button("Exit")]
    ]
    window = sg.Window("Rock Paper Scissors", layout, font=("Arial",14), enable_close_attempted_event=True, return_keyboard_events=True, finalize=True)
    default_color = sg.theme_button_color()
    

    def reset_color() -> None:
        '''Resets the button colors.'''

        window["-USER_ROCK-"].update(button_color=default_color)
        window["-USER_PAPER-"].update(button_color=default_color)
        window["-USER_SCISSORS-"].update(button_color=default_color)
        window["-CPU_ROCK-"].update(button_color=default_color)
        window["-CPU_PAPER-"].update(button_color=default_color)
        window["-CPU_SCISSORS-"].update(button_color=default_color)


    def update_calculations(cpu_result: np.ndarray | list[list[str]]):
        '''Updates the odd calculations visualization from the nn for each option.'''

        window["-CPU_ROCK_CALC-"].update(cpu_result[0][0])
        window["-CPU_PAPER_CALC-"].update(cpu_result[1][0])
        window["-CPU_SCISSORS_CALC-"].update(cpu_result[2][0])


    def reset_event() -> SimpleNeuralNetwork:
        '''Resets all values on the window and returns a new nn.'''

        reset_color()
        update_calculations([[""],[""],[""]])
        window["-CPU_SCORE-"].update(0)
        window["-USER_SCORE-"].update(0)
        window["-TIES-"].update(0)
        return SimpleNeuralNetwork(3, hidden_nodes, 3, hidden_layers, "tanh")


    nn = SimpleNeuralNetwork(3, hidden_nodes, 3, hidden_layers, "tanh")
    
    while True:
        event, values = window.read()
        reset_color()
        match event:
            case sg.WIN_CLOSE_ATTEMPTED_EVENT | "Exit":
                break
            case "-USER_ROCK-" | "r":
                cpu_result = nn.train([1,-1,-1], [-1,1,-1], learning_rate)
                update_calculations(np.round(cpu_result, 3))
                if np.argmax(cpu_result) == 2:  # Win.
                    window["-USER_ROCK-"].update(button_color="green")
                    window["-CPU_SCISSORS-"].update(button_color="red")                   
                    window["-USER_SCORE-"].update(int(values["-USER_SCORE-"]) + 1)
                elif np.argmax(cpu_result) == 1:  # Lose.
                    window["-USER_ROCK-"].update(button_color="red")
                    window["-CPU_PAPER-"].update(button_color="green")
                    window["-CPU_SCORE-"].update(int(values["-CPU_SCORE-"]) + 1)
                else:  # Tie.
                    window["-USER_ROCK-"].update(button_color="blue")
                    window["-CPU_ROCK-"].update(button_color="blue")
                    window["-TIES-"].update(int(values["-TIES-"]) + 1)
            case "-USER_PAPER-" | "p":
                cpu_result = nn.train([-1,1,-1], [-1,-1,1], learning_rate)
                update_calculations(np.round(cpu_result, 3))
                if np.argmax(cpu_result) == 0:  # Win.
                    window["-USER_PAPER-"].update(button_color="green")
                    window["-CPU_ROCK-"].update(button_color="red")                   
                    window["-USER_SCORE-"].update(int(values["-USER_SCORE-"]) + 1)
                elif np.argmax(cpu_result) == 2:  # Lose.
                    window["-USER_PAPER-"].update(button_color="red")
                    window["-CPU_SCISSORS-"].update(button_color="green")
                    window["-CPU_SCORE-"].update(int(values["-CPU_SCORE-"]) + 1)
                else:  # Tie.
                    window["-USER_PAPER-"].update(button_color="blue")
                    window["-CPU_PAPER-"].update(button_color="blue")
                    window["-TIES-"].update(int(values["-TIES-"]) + 1)
            case "-USER_SCISSORS-" | "s":  # Win.
                window["-USER_SCISSORS-"].update(button_color="green")
                cpu_result = nn.train([-1,-1,1], [1,-1,-1], learning_rate)
                update_calculations(np.round(cpu_result, 3))
                if np.argmax(cpu_result) == 1:  # Lose.
                    window["-USER_SCISSORS-"].update(button_color="green")
                    window["-CPU_PAPER-"].update(button_color="red")
                    window["-USER_SCORE-"].update(int(values["-USER_SCORE-"]) + 1)
                elif np.argmax(cpu_result) == 0:
                    window["-USER_SCISSORS-"].update(button_color="red")
                    window["-CPU_ROCK-"].update(button_color="green")
                    window["-CPU_SCORE-"].update(int(values["-CPU_SCORE-"]) + 1)
                else:  # Tie.
                    window["-USER_SCISSORS-"].update(button_color="blue")
                    window["-CPU_SCISSORS-"].update(button_color="blue")
                    window["-TIES-"].update(int(values["-TIES-"]) + 1)
            case "Reset":
                nn = reset_event()

    window.close()


if __name__ == "__main__":
    main()
    