def a_star(graph, start, goal,heuristc):
  open_list = [start]
  closed_list=[]
  g_cost={start:0}
  parent={}
  while open_list:
    current= min(open_list,key=lambda x:g_cost[x]+heuristc[x])
    if current==goal:
      path=[]
      while current in parent:
        path.append(current)
        current=parent[current]
      path.append(start)
      return path[::-1]
    open_list.remove(current)
    closed_list.append(current)

    for neighbor, cost in graph[current]:
      if neighbor in closed_list:
        continue
      tentative_g=g_cost[current]+cost
      if neighbor not in open_list:
        open_list.append(neighbor)
      elif tentative_g>=g_cost.get(neighbor,float('inf')):
        continue
      parent[neighbor]=current
      g_cost[neighbor]=tentative_g
  return None
    
graph = {
    'A': [('B', 1), ('C', 3)],
    'B': [('D', 3), ('E', 1)],
    'C': [('F', 5)],
    'D': [('G', 3)],
    'E': [('G', 1)],
    'F': [('G', 2)],
    'G': []
}

heuristic = {
    'A': 7, 
    'B': 6, 
    'C': 5, 
    'D': 3, 
    'E': 1, 
    'F': 2, 
    'G': 0
}
print(a_star(graph,'A','G',heuristic))