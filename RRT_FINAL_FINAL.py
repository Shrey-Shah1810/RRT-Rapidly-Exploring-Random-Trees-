# from RRT_connect import Curr_nodes
import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt
import time

def signum(i):
    if(i>0):
        return 1
    else: return -1

def resize_frame(frame,scale):
    resized_dimensions=(int(frame.shape[0]*scale),int(frame.shape[1]*scale))
    return cv.resize(frame,resized_dimensions,interpolation=cv.INTER_AREA)
STARTCOLOR=(27,216,17)
ENDCOLOR=(13,13,243)
Obs_Color=(207,226,243)
img=cv.imread("photo2.PNG")

img=resize_frame(img,2)
img2=img.copy()
Shape=(img.shape[1],img.shape[0]) ##Shape tuple first element is Columns(Width) Second is Rows(Height)
# plt.imshow(img)
# plt.show()
#SNodes and ENodes store the list of the start and end nodes as well as their diameters 
SNodes=[]
ENodes=[]

Curr_nodes_st=[]
Curr_nodes_end=[]
def change_weight(li,change):
    if(li==None):
        return
    for i in li:
        i.weight=i.weight-change
        change_weight(i.NextNodes,change)

class Node:
 
    def __init__(self,x1,y1):
        
        self.x=x1
        self.y=y1
        self.PrevNode=None
        self.PrevEdge=None
        # self.NextNodes=None ##List
        self.NextNodes=[]
        self.weight=0
        self.type=0
    def isTraversable(self):
        if(abs(img[self.y,self.x][0]-Obs_Color[0])<50 and abs(img[self.y,self.x][1]-Obs_Color[1])<50 and abs(img[self.y,self.x][2]-Obs_Color[2])<50):
            return False
        else: return True
    
    def colorNode(self,color):
        cv.rectangle(img,(self.x-2,self.y-2),(self.x+2,self.y+2),color,thickness=-1)


    
    def isEndNode(self):
        if(((self.x-CENTRE_END[1])**2+(self.y-CENTRE_END[0])**2)<=(dE/2)**2):
            return True
        else :return False
    def isStartNode(self):
        if(abs(self.x-CENTRE_START[1])<(dS/2) and abs(self.y-CENTRE_START[0])<(dS/2)):
            return True
        else :return False

    def Compare(self,node):
        if self==None or node==None:
            return False
        else :
            return ((self.x==node.x) and (self.y==node.y))

    def isTraversable2(self,Node):
        Bol=True 
        x_curr=self.x
        y_curr=self.y
        if not (Node.x==self.x):
            const=abs((Node.y-self.y)/(Node.x-self.x))
        while (not (x_curr==Node.x and y_curr==Node.y)):
            if(abs(img[y_curr,x_curr][0]-Obs_Color[0])<70 and abs(img[y_curr,x_curr][1]-Obs_Color[1])<70 and abs(img[y_curr,x_curr][2]-Obs_Color[2])<70):
                Bol=False 
                break 
            if(Node.x==x_curr):
                y_curr+=signum((Node.y-y_curr))
                continue 
            v=abs((Node.y-self.y)/(Node.x-self.x))
            if(v>const):
                y_curr+=signum(Node.y-y_curr)
            else:
                x_curr+=signum(Node.x-x_curr)
        # if(Bol==False):
            # print("No")
        return Bol 


class Edge:
    def __init__(self,Node1,Node2):
        self.edge =[Node1,Node2]
    def colorEdge(self,color,tcc):
        Scord=(self.edge[0].x,self.edge[0].y)
        Ecord=(self.edge[1].x,self.edge[1].y)
        cv.line(img, Scord, Ecord, color, thickness=tcc)

    def Compare(self,Ed):

    	return ((self.edge[0].x==Ed.edge[0].x and self.edge[0].y==Ed.edge[0] and self.edge[1].x==Ed.edge[1].x and self.edge[1].y==Ed.edge[1]) or (self.edge[1].x==Ed.edge[0].x and self.edge[1].y==Ed.edge[0] and self.edge[0].x==Ed.edge[1].x and self.edge[0].y==Ed.edge[1]))


# STARTCOLOR=(27,216,17)
# ENDCOLOR=(13,13,243)
# img=cv.imread("photo1.PNG")

def resize_frame(frame,scale):
    resized_dimensions=(int(frame.shape[0]*scale),int(frame.shape[1]*scale))
    return cv.resize(frame,resized_dimensions,interpolation=cv.INTER_AREA)

def distance(x1,y1,x2,y2):
	return ((x2-x1)**2+(y2-y1)**2)**0.5
 
# Curr_nodes=[]
sT=0
dm=0
# Bol=False
# dm=0
for i in range(0,Shape[1]):
    for j in range(0,Shape[0]):
        if(abs(img2[i,j][0]-STARTCOLOR[0])<30) and (abs(img2[i,j][1]-STARTCOLOR[1])<30) and (abs(img2[i,j][2]-STARTCOLOR[2])<30):
            sT=i
            dm=0

            while(abs(img2[sT,j][0]-STARTCOLOR[0])<50) and (abs(img2[sT,j][1]-STARTCOLOR[1])<50) and (abs(img2[sT,j][2]-STARTCOLOR[2])<50):
                dm+=1
                sT+=1

            tup=(j,int((sT+i-1)/2),dm)
            SNodes.append(tup)
            node=Node(tup[0],tup[1])
            node.type=1
            Curr_nodes_st.append(node)
            # print(tup[0],tup[1])
            cv.circle(img2,(int(tup[0]),int(tup[1])),int((dm+10)/2),(0,0,0),thickness=-1)
            # print(f"dm start={dm}")
            # cv.imshow("image",img2)
            # cv.waitKey(0)
            # time.sleep(100)

        if (abs(img2[i,j][0]-ENDCOLOR[0])<30) and (abs(img2[i,j][1]-ENDCOLOR[1])<30) and (abs(img2[i,j][2]-ENDCOLOR[2])<30):
            sT=i 
            dm=0
            while (abs(img2[sT,j][0]-ENDCOLOR[0])<30) and (abs(img2[sT,j][1]-ENDCOLOR[1])<30) and (abs(img2[sT,j][2]-ENDCOLOR[2])<30):
                dm+=1
                sT+=1
            tupl=(j,int((sT+i-1)/2),dm)
            node=Node(tupl[0],tupl[1])
            node.type=2
            Curr_nodes_end.append(node)
            cv.circle(img2,(int(tupl[0]),int(tupl[1])),int((dm+10)/2),(0,0,0),thickness=-1)
            
dS=tup[2]
dE=tupl[2]

inp=int(input("Enter start node :\n1.Top left\n2.Top Right\n3.Bottom Left\nInput:  "))
StartNode=Curr_nodes_end[inp-1]
inp=int(input("Enter End node :\n1.Top Right\n2.Bottom Right\n3.Bottom Left\nInput:  "))
EndNode=Curr_nodes_st[inp-1]
CENTRE_END=(EndNode.y,EndNode.x)
CENTRE_START=(StartNode.y,StartNode.x)
# print(CENTRE_START)

class rrtstar:
    
    def __init__(self,Start,EndNode,Iter,Step_Len,prob_value,radius):
        
        
        self.StartNode=Start
        self.StartNode.type=1
        self.curr_list_st=[self.StartNode]
        self.End=EndNode
        self.End.type=2
        self.curr_list_end=[self.End]
        self.length=Step_Len
        self.NodeSets=[[None for i in range(Shape[0])] for j in range(Shape[1])]   #---> alag se in RRT*  
        self.NodeSets[self.StartNode.y][self.StartNode.x]=self.StartNode 
        self.NodeSets[self.End.y][self.End.x]=self.End        
        self.prob_num=prob_value    
        self.Radius=radius
        self.iter=Iter       
 
    def solution(self):
        SolEdges=[]
        # shrey=False
        kushaz=self.prob_num
        SolWeight=1000000
        counter=2*self.iter
        
        # for i in range(0,self.iter):
        while(counter>0):
            shrey=False
            
            p_st=random.randint(0,9)
            if(p_st<kushaz):
                Rand=(random.randint(CENTRE_END[1]-20,CENTRE_END[1]+20),random.randint(CENTRE_END[0]-20,CENTRE_END[0]+20))
            else :
                Rand=(random.randint(0,Shape[0]-1),random.randint(0,Shape[1]-1))
            MinDist=1000000
            ClosestNode=None
            X=0
            Y=0
            for j in self.curr_list_st:
                PresentDist=((Rand[0]-j.x)**2+(Rand[1]-j.y)**2)**0.5
                if PresentDist<MinDist:
                    ClosestNode=j
                    MinDist=PresentDist
            # print(ClosestNode.x," ",ClosestNode.y)
            if MinDist:
                X=int(ClosestNode.x+int(self.length*(Rand[0]-ClosestNode.x)/MinDist))
                Y=int(ClosestNode.y+int(self.length*(Rand[1]-ClosestNode.y)/MinDist))
                if X>=Shape[0]:
                    X=(Shape[0]-1)
                elif X<0:
                    X=0
                if Y>=Shape[1]:
                    Y=(Shape[1]-1)
                elif Y<0:
                    Y=0
            
                # print(X," ",Y)
                NewNode=Node(X,Y)
                NewNode.type=1
                
                # OtherNode=None
                #Adding parent node to NewNode
                #shrey is false
                if(Node.isTraversable(NewNode) and self.NodeSets[NewNode.y][NewNode.x]==None):
                    shrey=True  
                    x_coord=None 
                    y_coord=None 
                    minimum=10000
                    ## TASK 1-FINDING BEST POSSIBLE PARENT NODE
                    for j in range(-self.Radius,self.Radius+1):
                        for k in range((-self.Radius),int((self.Radius+1))):
                            if(j+NewNode.y<Shape[1] and k+NewNode.x<Shape[0] and j+NewNode.y>=0 and k+NewNode.x>=0):
                                if(self.NodeSets[j+NewNode.y][k+NewNode.x]!=None and self.NodeSets[j+NewNode.y][k+NewNode.x].type==1):
                                    if(minimum>distance(NewNode.x,NewNode.y,k+NewNode.x,j+NewNode.y)+self.NodeSets[j+NewNode.y][k+NewNode.x].weight and Node.isTraversable2(NewNode,self.NodeSets[j+NewNode.y][k+NewNode.x])):
                                        minimum=distance(NewNode.x,NewNode.y,k+NewNode.x,j+NewNode.y)+self.NodeSets[j+NewNode.y][k+NewNode.x].weight
                                        x_coord=k+NewNode.x
                                        y_coord=j+NewNode.y

                    if(x_coord==None or y_coord==None):
                        # continue
                        shrey=False
                        # print(i,shrey)

                if(shrey==True):
                    counter-=1
                    # print(i,shrey)
                    Node.colorNode(NewNode,(0,255,255))
                    self.curr_list_st.append(NewNode)
                    NewNode.weight=minimum
                    self.NodeSets[NewNode.y][NewNode.x]=NewNode
                    self.NodeSets[y_coord][x_coord].NextNodes.append(NewNode)
                    NewNode.PrevNode=(self.NodeSets[y_coord][x_coord])
                    t=Edge(NewNode,NewNode.PrevNode)
                    Edge.colorEdge(t,(60,20,220),2)
	 				
                		#I feel the love and I feel it burn down this river, every turn, Hope is r 4 letter word 
                    ##TASK 2-FINDING IF THIS NEW NODE CAN BE A BETTER PARENT NODE FOR OTHER NODES
                    for j in range(-self.Radius,self.Radius+1):
                        for k in range((-self.Radius),int((self.Radius+1))):
                            if(j+NewNode.y<Shape[1] and k+NewNode.x<Shape[0] and j+NewNode.y>=0 and k+NewNode.x>=0):
                                if(self.NodeSets[j+NewNode.y][k+NewNode.x]!=None and self.NodeSets[j+NewNode.y][k+NewNode.x].type==1):
                                    if((NewNode.weight+distance(NewNode.x,NewNode.y,NewNode.x+k,NewNode.y+j))<self.NodeSets[j+NewNode.y][k+NewNode.x].weight and Node.isTraversable2(NewNode,self.NodeSets[j+NewNode.y][k+NewNode.x])):
                                        previous=self.NodeSets[j+NewNode.y][k+NewNode.x].PrevNode
                                        change=self.NodeSets[j+NewNode.y][k+NewNode.x].weight-(NewNode.weight+distance(NewNode.x,NewNode.y,NewNode.x+k,NewNode.y+j))

                                        for l in range(0,len(previous.NextNodes)):
                                            if(self.NodeSets[j+NewNode.y][k+NewNode.x].PrevNode.NextNodes[l].x==(k+NewNode.x)) and (self.NodeSets[j+NewNode.y][k+NewNode.x].PrevNode.NextNodes[l].y==(j+NewNode.y)):
                                                self.NodeSets[j+NewNode.y][k+NewNode.x].PrevNode.NextNodes.pop(l)
                                                break 

                                        self.NodeSets[j+NewNode.y][k+NewNode.x].PrevNode=NewNode
                                        self.NodeSets[j+NewNode.y][k+NewNode.x].weight=(NewNode.weight+distance(NewNode.x,NewNode.y,NewNode.x+k,NewNode.y+j))
                                        NewNode.NextNodes.append(self.NodeSets[j+NewNode.y][k+NewNode.x])
                                        change_weight(self.NodeSets[j+NewNode.y][k+NewNode.x].NextNodes,change)
                                        color=(60,20,220)
                                        tc=2
                                        for l in SolEdges:
                                            if Edge.Compare(l,Edge(self.NodeSets[j+NewNode.y][k+NewNode.x].PrevNode,previous)):
                                                color=(245,105,65)
                                                tc=4
                                        Edge.colorEdge(Edge(NewNode,self.NodeSets[j+NewNode.y][k+NewNode.x]),color,tc)
                                        Edge.colorEdge(Edge(previous,self.NodeSets[j+NewNode.y][k+NewNode.x]),(0,0,0),tc)

                    min_x=None
                    min_y=None                 
                    for j in range(-self.Radius,self.Radius+1):
                        for k in range((-self.Radius),int((self.Radius+1))):
                            if(j+NewNode.y<Shape[1] and k+NewNode.x<Shape[0] and j+NewNode.y>=0 and k+NewNode.x>=0):
                                if(self.NodeSets[j+NewNode.y][k+NewNode.x]!=None and self.NodeSets[j+NewNode.y][k+NewNode.x].type==2):
                                    if(SolWeight>(NewNode.weight+self.NodeSets[j+NewNode.y][k+NewNode.x].weight+distance(NewNode.x,NewNode.y,NewNode.x+k,NewNode.y+j)) and Node.isTraversable2(NewNode,self.NodeSets[NewNode.y+j][NewNode.x+k])):
                                        SolWeight=(NewNode.weight+self.NodeSets[j+NewNode.y][k+NewNode.x].weight+distance(NewNode.x,NewNode.y,NewNode.x+k,NewNode.y+j))
                                        kushaz=-1
                                        min_x=k+NewNode.x
                                        min_y=j+NewNode.y
                                       

                    if(min_x!=None):
                        ConnecT=Edge(self.NodeSets[min_y][min_x],NewNode)
                        Edge.colorEdge(ConnecT,(245,105,65),4)
                        for i in SolEdges:
                            Edge.colorEdge(i,(0,0,0),4)
                            Edge.colorEdge(i,(60,20,220),2)

                        SolEdges.clear()
                        SolEdges.append(ConnecT)
                        PresentNode1=NewNode

                        while(not(PresentNode1.PrevNode==None)):
                            # print("Hlo")
                            Dedge1=Edge(PresentNode1.PrevNode,PresentNode1)
                            SolEdges.append(Dedge1)
                            Edge.colorEdge(Dedge1,(245,105,65),4)
                            PresentNode1=PresentNode1.PrevNode

                        PresentNode2=self.NodeSets[min_y][min_x]
                        while(not(PresentNode2.PrevNode==None)):
                            Dedge2=Edge(PresentNode2.PrevNode,PresentNode2)
                            SolEdges.append(Dedge2)
                            Edge.colorEdge(Dedge2,(245,105,65),4)
                            PresentNode2=PresentNode2.PrevNode


            p_st=random.randint(0,9)
            if(p_st<kushaz):
                Rand=(random.randint(CENTRE_START[1]-20,CENTRE_START[1]+20),random.randint(CENTRE_START[0]-20,CENTRE_START[0]+20))
            else :
                Rand=(random.randint(0,Shape[0]-1),random.randint(0,Shape[1]-1))
            MinDist=100000 
            ClosestNode=None
            X=0
            Y=0
            for j in self.curr_list_end:
                PresentDist=((Rand[0]-j.x)**2+(Rand[1]-j.y)**2)**0.5
                if PresentDist<MinDist:
                    ClosestNode=j
                    MinDist=PresentDist
            if MinDist:
                X=int(ClosestNode.x+int(self.length*(Rand[0]-ClosestNode.x)/MinDist))
                Y=int(ClosestNode.y+int(self.length*(Rand[1]-ClosestNode.y)/MinDist))
                if X>=Shape[0]:
                    X=(Shape[0]-1)
                elif X<0:
                    X=0
                if Y>=Shape[1]:
                    Y=(Shape[1]-1)
                elif Y<0:
                    Y=0
           

                NewNode=Node(X,Y)
                NewNode.type=2
                
                # OtherNode=None
                #Adding parent node to NewNode
                if(Node.isTraversable(NewNode) and self.NodeSets[NewNode.y][NewNode.x]==None):
                    
                    
                    x_coord=None 
                    y_coord=None 
                    minimum=10000
                    ## TASK 1-FINDING BEST POSSIBLE PARENT NODE
                    # print(NewNode.x,NewNode.y)
                    for j in range(-self.Radius,self.Radius+1):
                        for k in range((-self.Radius),int((self.Radius+1))):
                            if(j+NewNode.y<Shape[1] and k+NewNode.x<Shape[0] and j+NewNode.y>=0 and k+NewNode.x>=0):
                                if(self.NodeSets[j+NewNode.y][k+NewNode.x]!=None and self.NodeSets[j+NewNode.y][k+NewNode.x].type==2):
                                    if(minimum>distance(NewNode.x,NewNode.y,k+NewNode.x,j+NewNode.y)+self.NodeSets[j+NewNode.y][k+NewNode.x].weight and Node.isTraversable2(NewNode,self.NodeSets[j+NewNode.y][k+NewNode.x])):
                                        minimum=distance(NewNode.x,NewNode.y,k+NewNode.x,j+NewNode.y)+self.NodeSets[j+NewNode.y][k+NewNode.x].weight
                                        x_coord=k+NewNode.x
                                        y_coord=j+NewNode.y
                    if(x_coord==None):
                        continue 
                    counter-=1
                    Node.colorNode(NewNode,(0,255,255))
                    self.curr_list_end.append(NewNode)
                    NewNode.weight=minimum
                    self.NodeSets[NewNode.y][NewNode.x]=NewNode
                    self.NodeSets[y_coord][x_coord].NextNodes.append(NewNode)
                    NewNode.PrevNode=(self.NodeSets[y_coord][x_coord])
                    Edge.colorEdge((Edge(NewNode,NewNode.PrevNode)),(0,255,0),2)
	 				
                		#I feel the love and I feel it burn down this river, every turn, Hope is r 4 letter word 
                    ##TASK 2-FINDING IF THIS NEW NODE CAN BE A BETTER PARENT NODE FOR OTHER NODES
                    for j in range(-self.Radius,self.Radius+1):
                        for k in range((-self.Radius),int((self.Radius+1))):
                            if(j+NewNode.y<Shape[1] and k+NewNode.x<Shape[0] and j+NewNode.y>=0 and k+NewNode.x>=0):
                                if(self.NodeSets[j+NewNode.y][k+NewNode.x]!=None and self.NodeSets[j+NewNode.y][k+NewNode.x].type==2):
                                    if((NewNode.weight+distance(NewNode.x,NewNode.y,NewNode.x+k,NewNode.y+j))<self.NodeSets[j+NewNode.y][k+NewNode.x].weight and Node.isTraversable2(NewNode,self.NodeSets[j+NewNode.y][k+NewNode.x])):
                                        previous=self.NodeSets[j+NewNode.y][k+NewNode.x].PrevNode
                                        change=self.NodeSets[j+NewNode.y][k+NewNode.x].weight-(NewNode.weight+distance(NewNode.x,NewNode.y,NewNode.x+k,NewNode.y+j))

                                        for l in range(0,len(previous.NextNodes)):
                                            if(j+NewNode.y<Shape[1] and k+NewNode.x<Shape[0] and j+NewNode.y>=0 and k+NewNode.x>=0 and self.NodeSets[j+NewNode.y][k+NewNode.x].PrevNode.NextNodes[l].x==(k+NewNode.x)) and (self.NodeSets[j+NewNode.y][k+NewNode.x].PrevNode.NextNodes[l].y==(j+NewNode.y)):
                                                self.NodeSets[j+NewNode.y][k+NewNode.x].PrevNode.NextNodes.pop(l)
                                                break 

                                        self.NodeSets[j+NewNode.y][k+NewNode.x].PrevNode=NewNode
                                        self.NodeSets[j+NewNode.y][k+NewNode.x].weight=(NewNode.weight+distance(NewNode.x,NewNode.y,NewNode.x+k,NewNode.y+j))
                                        NewNode.NextNodes.append(self.NodeSets[j+NewNode.y][k+NewNode.x])
                                        change_weight(self.NodeSets[j+NewNode.y][k+NewNode.x].NextNodes,change)
                                        color=(0,255,0)
                                        tc=2
                                        for l in SolEdges:
                                            if Edge.Compare(l,Edge(self.NodeSets[j+NewNode.y][k+NewNode.x].PrevNode,previous)):
                                                color=(245,105,65)
                                                tc=4
                                        Edge.colorEdge(Edge(NewNode,self.NodeSets[j+NewNode.y][k+NewNode.x]),color,tc)
                                        Edge.colorEdge(Edge(previous,self.NodeSets[j+NewNode.y][k+NewNode.x]),(0,0,0),tc)

                    min_x=None
                    min_y=None 
                    for j in range(-self.Radius,self.Radius+1):
                        for k in range((-self.Radius),int((self.Radius+1))):
                            if(j+NewNode.y<Shape[1] and k+NewNode.x<Shape[0] and j+NewNode.y>0 and k+NewNode.x>0):
                                if(self.NodeSets[j+NewNode.y][k+NewNode.x]!=None and self.NodeSets[j+NewNode.y][k+NewNode.x].type==1):
                                    if(SolWeight>(NewNode.weight+self.NodeSets[j+NewNode.y][k+NewNode.x].weight+distance(NewNode.x,NewNode.y,NewNode.x+k,NewNode.y+j)) and Node.isTraversable2(NewNode,self.NodeSets[NewNode.y+j][NewNode.x+k])):
                                        kushaz=-1
                                        min_x=k+NewNode.x
                                        min_y=j+NewNode.y
                                        SolWeight=(NewNode.weight+self.NodeSets[j+NewNode.y][k+NewNode.x].weight+distance(NewNode.x,NewNode.y,NewNode.x+k,NewNode.y+j))
                                       

                    if(min_x!=None):
                        ConnecT=Edge(self.NodeSets[min_y][min_x],NewNode)
                        Edge.colorEdge(ConnecT,(245,105,65),4)
                        for i in SolEdges:

                            Edge.colorEdge(i,(0,0,0),4)
                            Edge.colorEdge(i,(0,255,0),2)

                        SolEdges.clear()
                        SolEdges.append(ConnecT)
                        PresentNode1=NewNode

                        while(not(PresentNode1.PrevNode==None)):
                            # print("Hlo")
                            Dedge1=Edge(PresentNode1.PrevNode,PresentNode1)
                            SolEdges.append(Dedge1)
                            Edge.colorEdge(Dedge1,(245,105,65),4)
                            PresentNode1=PresentNode1.PrevNode


                        PresentNode2=self.NodeSets[min_y][min_x]  
                        while(not(PresentNode2.PrevNode==None)):
                            Dedge2=Edge(PresentNode2.PrevNode,PresentNode2)
                            SolEdges.append(Dedge2)
                            Edge.colorEdge(Dedge2,(245,105,65),4)
                            PresentNode2=PresentNode2.PrevNode


                    #After this we need to check in the circle for a opp(green or .type=2) and make bonds and copy for red then backtracking and then chill
#--------------------------------------------Done Till here------------------------------------------------------------------------------------------------------------------------------------------------------------------
                
    # What doesn't kills you makes you stronger..................
    #Sharp edges have concequences 

                cv.imshow("image",img)
                cv.waitKey(1)
        for i in SolEdges:
            Edge.colorEdge(i,(245,105,65),5)
        cv.imshow("image",img)
        cv.waitKey(0)

# def __init__(self,Start,EndNode,Iter,Step_Len,prob_value,radius):
ITERATIONS=int(input("NUMBER OF ITERATIONS : "))
# STARTNODE=Node(CENTRE_START[1],CENTRE_START[0])

RRT = rrtstar(StartNode,EndNode,ITERATIONS,24,2,24)
rrtstar.solution(RRT)




