from tkinter import *
import tkinter.messagebox as messagebox
import pymysql
from tkinter import ttk

root = Tk()
root.geometry('500x400')
root.title('MySQL GUI')

#tabs
tab_parent = ttk.Notebook(root)
tab1 = ttk.Frame(tab_parent)
tab2 = ttk.Frame(tab_parent)
tab3 = ttk.Frame(tab_parent)
tab4 = ttk.Frame(tab_parent)
tab5 = ttk.Frame(tab_parent)
tab6 = ttk.Frame(tab_parent)
tab_parent.add(tab1, text="sex")
tab_parent.add(tab2, text="country")
tab_parent.add(tab3, text="roles")
tab_parent.add(tab4, text="submitted_photos")
tab_parent.add(tab5, text="user_information")
tab_parent.add(tab6, text="users")
tab_parent.pack(expand=1, fill='both')

#################    tab2  == country  #################################
#functions
def insert_country():
    id_country = e_id_country.get()
    country = e_country.get()
    if (id_country == '' or country == ''):
        messagebox.showinfo('insert status', 'all fields are required!')
    else:
        con = pymysql.connect(host='localhost', user='root', password='root', database='mydb')
        cursor = con.cursor()
        cursor.execute("insert into country values('" + id_country + "','" + country + "')")
        cursor.execute("commit")
        e_id_country.delete(0, 'end')
        e_country.delete(0, 'end')
        show_country()
        messagebox.showinfo("insert status", "inserted successfully")
        con.close()


def delete_country():
    if (e_id_country.get() == ''):
        messagebox.showinfo('Delete status', ' id is compolsary for delete function')
    else:
        con = pymysql.connect(host='localhost', user='root', password='root', database='mydb')
        cursor = con.cursor()
        cursor.execute("delete from country where id='" + e_id_country.get() + "'")
        cursor.execute("commit")

        e_id_country.delete(0, 'end')
        e_country.delete(0, 'end')
        show_country()
        messagebox.showinfo("delete status", "deleted successfully")
        con.close()


def update_country():
    id_country = e_id_country.get()
    country = e_country.get()
    if (id_country == '' or country == ''):
        messagebox.showinfo('update status', 'all fields are required!')
    else:
        con = pymysql.connect(host='localhost', user='root', password='root', database='mydb')
        cursor = con.cursor()
        cursor.execute("update country set country='" + country + "' where id ='" + id_country + "'")
        cursor.execute("commit")

        e_id_country.delete(0, 'end')
        e_country.delete(0, 'end')
        show_country()
        messagebox.showinfo("update status", "updated successfully")
        con.close()


def get_country():
    if (e_id_country.get() == ''):
        messagebox.showinfo('fetch status', ' id is compolsary for delete function')
    else:
        con = pymysql.connect(host='localhost', user='root', password='root', database='mydb')
        cursor = con.cursor()
        cursor.execute("select * from country where id='" + e_id_country.get() + "'")
        e_id_country.delete(0, 'end')
        e_country.delete(0, 'end')
        rows = cursor.fetchall()
        for row in rows:
            e_id_country.insert(0, row[0])
            e_country.insert(0, row[1])
        show_country()
        con.close()

#interactive list
def show_country():
    con = pymysql.connect(host='localhost', user='root', password='root', database='mydb')
    cursor = con.cursor()
    cursor.execute("select * from country")
    rows = cursor.fetchall()
    list_country.delete(0, list_country.size())
    for row in rows:
        insertdata = str(row[0]) + '      ' + row[1]
        list_country.insert(list_country.size() + 1, insertdata)
    con.close()

#Field names
id_country = Label(tab2, text='id', font=('bold', 10))
id_country.place(x=40, y=30)
country = Label(tab2, text='country', font=('bold', 10))
country.place(x=40, y=60)

#input fields
e_id_country = Entry(tab2)
e_id_country.place(x=100, y=30)
e_country = Entry(tab2)
e_country.place(x=100, y=60)

#buttons
insert_country = Button(tab2, text='insert', font=('bold', 10), bg='gray', command=insert_country)
insert_country.place(x=40, y=240)

delete_country = Button(tab2, text='delete', font=('bold', 10), bg='gray', command=delete_country)
delete_country.place(x=100, y=240)

update_country = Button(tab2, text='update', font=('bold', 10), bg='gray', command=update_country)
update_country.place(x=160, y=240)

get_country = Button(tab2, text='get', font=('bold', 10), bg='gray', command=get_country)
get_country.place(x=220, y=240)

#list of table content
list_country = Listbox(tab2)
list_country.place(x=300, y=10)
show_country()



## This code is for 1 tab for 1 table/ all the other tables in the database will have thei own tabs (as seen above) and for each tab you need to define new functions,
## new input fields, new buttons etc...
