import numpy as np
import matplotlib.pyplot as plt
from time import time

t0=time()

def normalize(vector):
    return vector / np.linalg.norm(vector,axis=1)[:,np.newaxis]

def reflected(vector, axis1):
    return vector - 2 * np.sum(vector*axis1,axis=1)[:,np.newaxis]*axis1

def sphere_intersect(center, radius, ray_origin, ray_direction):
    if len(ray_origin)==3:
        b = 2 * np.dot(ray_direction, ray_origin - center)
        c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    else:
        b=2 * np.sum(ray_direction*(ray_origin - center),axis=1)
        c = np.linalg.norm(ray_origin - center,axis=1) ** 2 - radius ** 2
    dist=np.zeros((len(ray_direction)))
    delta = b ** 2 - 4 * c  #REVISAR ESTA PARTE SI ALGO VA MAL JEJE
    t1 = (-b[delta>0] + np.sqrt(delta[delta>0])) / 2
    t2 = (-b[delta>0] - np.sqrt(delta[delta>0])) / 2
    dist[delta>0]=np.minimum(t1,t2)
    dist[delta<0]=None
    dist[dist<0]=None
    return dist

def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = [sphere_intersect(obj['center'], obj['radius'], ray_origin, ray_direction) for obj in objects]
    min_distance = np.inf
    min_dists=np.zeros((len(distances[0]),len(distances)))
    nearest_objs=np.zeros((len(distances[0]),len(distances)))
    for index, distance in enumerate(distances):
        min_dists[:,index][distances[index]<min_distance]=distances[index][distances[index]<min_distance]
        min_dists[:,index][np.bitwise_not(distances[index]<min_distance)]=min_distance
    obj_ind=np.argmin(min_dists,axis=1)
    min_distances=np.min(min_dists,axis=1)
    obj_ind[min_distances==np.inf]=len(objects)
    obj_ind=list(obj_ind)
    objects=np.array(objects+[None])
    objects=np.tile(objects,len(obj_ind))
    nearest_object=objects[obj_ind]
    return np.array(obj_ind),nearest_object, min_distances

def raytrace_frame(width,height,objects,light,screen,camera,max_depth):
    #ratio=float(width)/height
    image=np.zeros((height, width, 3))     #Imagen RGB

    y=np.linspace(screen[1], screen[3], height)
    x=np.linspace(screen[0], screen[2], width)
    z=np.zeros((len(x)))
    X,Y=np.meshgrid(x,y)
    XY=np.array([X.flatten(),Y.flatten()]).T
    XYZ=np.zeros((len(XY[:,0]),len(XY[0,:])+1))
    z=np.ones((len(XY[:,0])))
    XYZ[:,2]=z
    XYZ[:,0:2]=XY
    direction=normalize(XYZ-camera)

    centers=np.array([je['center'] for je in objects])
    ambients=np.array([je['ambient'] for je in objects])
    diffuses=np.array([je['diffuse'] for je in objects])
    speculars=np.array([je['specular'] for je in objects])
    shininesses=np.array([je['shininess'] for je in objects])
    reflections=np.array([je['reflection'] for je in objects])

    origin=camera
    color=np.zeros((width*height,3))
    reflection=np.ones((width*height))

    for k in range(max_depth):
        obj_ind,nearest_object, min_distance = nearest_intersected_object(objects, origin, direction)
        inter=nearest_object!=None
        intersection=origin+direction*min_distance[:,np.newaxis]
        normal_to_surface=np.ones(np.shape(intersection))*np.inf
        normal_to_surface[inter]=normalize(intersection[inter]-centers[obj_ind[inter]])
        shifted_point = intersection + 1e-5 * normal_to_surface
        intersection_to_light = normalize(light['position']-shifted_point)
        
        nu,_, min_distance = nearest_intersected_object(objects, shifted_point, intersection_to_light)
        intersection_to_light_distance = np.linalg.norm(light['position'] - intersection,axis=1)
        lighted=min_distance>intersection_to_light_distance
        illumination=np.zeros((len(lighted),3))
        illumination[lighted]+=ambients[obj_ind[lighted]]*light['ambient']
        illumination[lighted]+=diffuses[obj_ind[lighted]] * light['diffuse'] * np.sum(intersection_to_light[lighted]*normal_to_surface[lighted],axis=1)[:,np.newaxis]
        intersection_to_camera = normalize(camera - intersection)
        H = normalize(intersection_to_light + intersection_to_camera)
        illumination[lighted] += speculars[obj_ind[lighted]] * light['specular'] * (np.sum(normal_to_surface[lighted]*H[lighted],axis=1) ** (shininesses[obj_ind[lighted]] / 4))[:,np.newaxis]

        if k==0:
            color += reflection[:,np.newaxis] * illumination
            reflection[lighted] *= reflections[obj_ind[lighted]]
        else:
            color[color1] += reflection[color1][:,np.newaxis] * illumination
            reflection[color1[lighted]] *= reflections[obj_ind[lighted]]
            #print(np.unique(reflection[color1[lighted]]))
        if k>0:
            color1=color1[np.where(lighted==1)[0]]
        else:
            color1=np.where(lighted==1)[0]

        origin = shifted_point[lighted]
        direction = reflected(direction[lighted], normal_to_surface[lighted])

    color=np.reshape(color,(height,width,3))
    image=np.clip(color,0,1)
    return image
    
objects = [
    { 'center': np.array([-0.2, 0, -1]), 'radius': 0.7, 'ambient': np.array([0.1, 0, 0]), 'diffuse': np.array([0.7, 0, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
    { 'center': np.array([0.1, -0.3, 0]), 'radius': 0.1, 'ambient': np.array([0.1, 0, 0.1]), 'diffuse': np.array([0.7, 0, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
    { 'center': np.array([-0.3, 0, 0]), 'radius': 0.15, 'ambient': np.array([0, 0.1, 0]), 'diffuse': np.array([0, 0.6, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
    { 'center': np.array([0, -9000, 0]), 'radius': 9000 - 0.7, 'ambient': np.array([0.1, 0.1, 0.1]), 'diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 }
]

light = { 'position': np.array([5,5,5]), 'ambient': np.array([1,1,1]), 'diffuse': np.array([1, 1, 1]), 'specular': np.array([1, 1, 1]) }

width=300
height=200
ratio=float(width)/height
screen=(-1,1/ratio,1,-1/ratio)
camera=np.array([0,0,2])
max_depth=3

image=raytrace_frame(width,height,objects,light,screen,camera,max_depth)

plt.imsave('lalala.png', image)

t1=time()

print('Tu raytracer tarda %.2f s en ejecutarse'%(t1-t0))