import PySimpleGUI as sg

# Create a layout for the GUI
layout = [[sg.Text('Select your favorite color:')],
          [sg.Radio('Red', 'colors', default=True),
           sg.Radio('Green', 'colors'),
           sg.Radio('Blue', 'colors')],
          [sg.Text('Select your preferred programming languages:')],
          [sg.Checkbox('Python'),
           sg.Checkbox('Java'),
           sg.Checkbox('C++')],
          [sg.Text('Select your preferred operating system:')],
          [sg.Combo(['Windows', 'MacOS', 'Linux'], size=(20, 1))],
          [sg.Text('Enter a password:')],
          [sg.Input(password_char='*')],
          [sg.Text('Choose a file:')],
          [sg.Input(), sg.FileBrowse()],
          [sg.Submit(), sg.Cancel()]
         ]

# Create the window from the layout
window = sg.Window('Demo', layout)

# Run the event loop to process user input
while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Cancel'):
        break
    elif event == 'Submit':
        sg.popup(f'Selected color: {values[0]}\n'
                 f'Preferred languages: {values[1:]}\n'
                 f'Operating system: {values[4]}\n'
                 f'Password: {values[5]}\n'
                 f'File: {values[6]}')

# Close the window
window.close()
