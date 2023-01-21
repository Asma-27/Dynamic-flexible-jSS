import numpy as np
import os # on importe le OS pour pouvoir utiliser les fonctions necessaires comme chdir ...
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import random
from collections import deque #Un dèque est une structure de données de type conteneur et qui s’utilise à la fois comme une pile ou file 
from tensorflow.python.keras import models,layers
import tensorflow as tf #dédié au réseaux de neurones de Deep Learning
from Job_Shop import Situation
from tensorflow.python.keras.optimizer_v1 import Adam
from Instance_Generator import Processing_time,A,D,M_num,Op_num,J,O_num,J_num
import matplotlib.pyplot as plt





class DQN:
    def __init__(self,):
        self.Hid_Size = 30

        # ------------Hidden layer=5   30 nodes each layer--------------
        model = models.Sequential() #créer un modèle de réseau de neurones séquentiel 
        model.add(layers.InputLayer(input_shape=(7,)))
        model.add(layers.Dense(self.Hid_Size, name='l1'))
        model.add(layers.Dense(self.Hid_Size, name='l2'))
        model.add(layers.Dense(self.Hid_Size, name='l3'))
        model.add(layers.Dense(self.Hid_Size, name='l4'))
        model.add(layers.Dense(self.Hid_Size, name='l5'))
        model.add(layers.Dense(6, name='l6')) # 
        model.compile(loss='mse',
                      optimizer='adam')
        self.model = model

        #------------Q-network Parameters-------------
        self.act_dim=[1,2,3,4,5,6]                        
        self.obs_n=[0,0,0,0,0,0,0]                            
        self.gama = 0.95  
        self.global_step = 0
        self.update_target_steps = 200  
        self.target_model = self.model

        #-------------------Agent-------------------
        self.e_greedy=0.6
        self.e_greedy_decrement=0.0001
        self.L=40          #Number of training episodes L

        #---------------Replay Buffer---------------
        self.buffer=deque(maxlen=2000)
        self.Batch_size=10       # Batch Size of Samples to perform gradient descent

    def replace_target(self):
        self.target_model.get_layer(name='l1').set_weights(self.model.get_layer(name='l1').get_weights())
        self.target_model.get_layer(name='l2').set_weights(self.model.get_layer(name='l2').get_weights())
        self.target_model.get_layer(name='l3').set_weights(self.model.get_layer(name='l3').get_weights())
        self.target_model.get_layer(name='l4').set_weights(self.model.get_layer(name='l4').get_weights())
        self.target_model.get_layer(name='l5').set_weights(self.model.get_layer(name='l5').get_weights())
        self.target_model.get_layer(name='l6').set_weights(self.model.get_layer(name='l6').get_weights())

    def replay(self):# utilisée pour mettre à jour les poids du modèle en utilisant les données stockées dans le buffer d'expérience.
        if self.global_step % self.update_target_steps == 0:
            self.replace_target()
        # replay the history and train the model
        minibatch = random.sample(self.buffer, self.Batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                k=self.target_model.predict(next_state)
                target = (reward + self.gama *
                          np.argmax(self.target_model.predict(next_state)))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        self.global_step += 1

    def Select_action(self,obs):
        # obs=np.expand_dims(obs,0)
        if random.random()<self.e_greedy:
            act=random.randint(0,5)
        else:
            act=np.argmax(self.model.predict(obs))
        self.e_greedy = max(
            0.01, self.e_greedy - self.e_greedy_decrement)  # 
        return act

    def _append(self, exp):
        self.buffer.append(exp)

    def main(self,J_num, M_num, O_num, J, Processing_time, D, A):
        k = 0
        x=[]
        Total_tard=[]
        TR=[]
        for i in range(self.L):
            Total_reward = 0
            x.append(i+1)
            print('-----------------------Commence le ',i+1,'entrainement suivant ------------------------------')
            obs=[0 for i in range(7)]
            obs = np.expand_dims(obs, 0)
            done=False
            Sit = Situation(J_num, M_num, O_num, J, Processing_time, D, A)
            for i in range(O_num):
                k+=1
                # print(obs)
                at=self.Select_action(obs)
                # print(at)
                if at==0:
                    at_trans=Sit.rule1()
                if at==1:
                    at_trans=Sit.rule2()
                if at==2:
                    at_trans=Sit.rule3()
                if at==3:
                    at_trans=Sit.rule4()
                if at==4:
                    at_trans=Sit.rule5()
                if at==5:
                    at_trans=Sit.rule6()
                print('Ceci est le ',i,"étape du processus>>","executer l'action:",at,' ',"placer l'objet",at_trans[0],'metre en place sur la machine ',at_trans[1])
                Sit.scheduling(at_trans)
                obs_t=Sit.Features()
                if i==O_num-1:
                    done=True
                #obs = obs_t
                obs_t = np.expand_dims(obs_t, 0)
                # obs = np.expand_dims(obs, 0)
                # print(obs,obs_t)
                r_t = Sit.reward(obs[0][6],obs[0][5],obs_t[0][6],obs_t[0][5],obs[0][0],obs_t[0][0])
                self._append((obs,at,r_t,obs_t,done))
                if k>self.Batch_size:
                    # batch_obs, batch_action, batch_reward, batch_next_obs,done= self.sample()
                    self.replay()
                Total_reward+=r_t
                obs=obs_t
            total_tadiness=0
            Job=Sit.Jobs
            j=0
            for i in (Job) :
                j=+1
            print(j)    
            E=0
            K=[i for i in range(len(Job))]
            End=[]
            for Ji in range(len(Job)):
                End.append(max(Job[Ji].End))
                if max(Job[Ji].End)>D[Ji]:
                    total_tadiness+=abs(max(Job[Ji].End)-D[Ji])
            print('<<<<<<<<<-----------------total_tardiness:',total_tadiness,'------------------->>>>>>>>>>')
            Total_tard.append(total_tadiness)
            print('<<<<<<<<<-----------------reward:',Total_reward,'------------------->>>>>>>>>>')
            TR.append(Total_reward)
            plt.plot(K,End,color='y')
            plt.plot(K,D,color='r')
            plt.show()
        plt.plot(x,Total_tard)
        plt.show()
        return Total_reward


d=DQN()
d.main(J_num, M_num, O_num, J, Processing_time, D, A)


