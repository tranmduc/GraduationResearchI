from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import ttk
from Rbfc import Rbfc
import general_functions as gf
import pandas as pd
from tkinter import scrolledtext


def data(): 
    global filename
    filename = askopenfilename(title = "Select file")
    
    e0.delete(0, END)
    e0.insert(0, filename)
 
    global df
    df = pd.read_csv(filename)
    
    global minmax
    
    header = ""
    for title in df.columns:
        header += title
        header += ","
    
    header = header[:-1]
    
    box0.delete(0, END)
    box0.insert(END, header)

    for index, row in df.iterrows():
        box0.insert(index+1, list(row))

    cols = list(df.columns)
    
    global col_len
    
    col_len = len(cols) - 2
    
    df = df.drop(df.columns[0], axis=1)
    minmax = [ [min, max] for min, max in zip(df.min(), df.max()) ]
    df = df.drop(cols[col_len-1], axis=1)
    

def split():
    (data_set, targets) = gf.csv_to_data_set(filename)
    global train_set, test_set, train_targets, test_targets
    (train_set, test_set, train_targets, test_targets) = gf.split_data_set(data_set, targets, test_size=float(e1.get()))
    
    test_df = pd.DataFrame(test_set)
    
    header = ""
    for title in df.columns:
        header += title
        header += ","
    
    header = header[:-1]
    
    box1.delete(0, END)
    box1.insert(END, header)

    for index, row in test_df.iterrows():
        box1.insert(index+1, list(row))
        
        
def train():   
    global rbfc
    global fuzzy_set_size
    
    fuzzy_set_size = int(variable.get())
    
    rbfc = Rbfc(train_set, train_targets, col_len, fuzzy_set_size, minmax)

    box2.delete(0, END)
    rules = rbfc.show_rules()
    
    for i in range(len(rules)):
        box2.insert(i+1, rules[i])
        
def predict():
    predictions = rbfc.predict(test_set)
    box3.insert(0, gf.accuracy(predictions, test_targets))
           
gui = Tk()
gui.title('FUZZY RULE-BASED CLASSIFICATION')
gui.geometry('1000x750')

w2 = Label(gui, justify=LEFT, text="FUZZY RULE-BASED CLASSIFICATION")
w2.config(font=("Elephant", 25))
w2.grid(row=0, column=0, columnspan = 2, padx=100)

Button(gui,text='Select Data File', command=data, width = 20).grid(row=1, column=0)
e0 = Entry(gui,text='')
e0.grid(row=2, column=0)

box0 = Listbox(gui, width = 100)
box0.grid(row=1, column=1, rowspan = 2)

Button(gui, text='Generate Test Set', command=split, width = 20).grid(row=7, column=0)
box1 = Listbox(gui, width = 100)
box1.grid(row=5, column=1, rowspan=3)
genlb = Label(gui, text="Test Set")
genlb.grid(row=4, column=1)

Button(gui, text='Training', command=train, width = 20).grid(row=12, column=0)
box2 = Listbox(gui, width = 100)
box2.grid(row=10, column=1, rowspan = 3)
trainlb = Label(gui, text="Rules")
trainlb.grid(row=9, column=1)

Button(gui, text='Predict', command=predict, width = 20).grid(row=15, column=0)
acc = Label(gui, text="Accuracy")
acc.grid(row=14, column=1)
box3 = Listbox(gui, height = 1, width = 40)
box3.grid(row=15, column=1)

treeXScroll = ttk.Scrollbar(gui, orient=HORIZONTAL)
treeXScroll.configure(command=box0.xview)
box0.configure(xscrollcommand=treeXScroll.set)
treeXScroll.grid(column=1, row=3)

treeXScroll2 = ttk.Scrollbar(gui, orient=HORIZONTAL)
treeXScroll2.configure(command=box1.xview)
box1.configure(xscrollcommand=treeXScroll2.set)
treeXScroll2.grid(column=1, row=8)

treeXScroll3 = ttk.Scrollbar(gui, orient=HORIZONTAL)
treeXScroll3.configure(command=box2.xview)
box2.configure(xscrollcommand=treeXScroll3.set)
treeXScroll3.grid(column=1, row=13)

ratio = DoubleVar(gui)
e1 = ttk.Entry(gui,textvariable = ratio)
e1.grid(row=6, column=0)

OPTIONS = ["3", "5", "7"]

variable = StringVar(gui)
variable.set(OPTIONS[0]) # default value

w = OptionMenu(gui, variable, *OPTIONS)
w.grid(column=0, row = 11)

size = Label(gui, text="Number of fuzzy set")
size.grid(row=10, column=0)

split = Label(gui, text="Test - Train Ratio")
split.grid(row=5, column=0)

gui.mainloop()