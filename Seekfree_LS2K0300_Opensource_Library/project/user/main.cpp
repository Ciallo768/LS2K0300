#include "zf_common_headfile.hpp"
//中断回调函数
uint8 Tcp_buffer[128];
zf_driver_tcp_client tcp_client_dev;
int frame_count = 0;                                      // 累计帧数
auto start_time = std::chrono::steady_clock::now();       // 统计起始时间
auto last_print_time = start_time;                        // 上一次打印帧率的时间
#define PRINT_INTERVAL 1000           // 帧率打印间隔（毫秒），即每秒打印一次

int32_t frame_cnt = 0;
void pit_callback()
{ 

    ICM_getEulerianAngles();
    if(it_time%10==0)
    dl1x_distance_raw = dl1x_dev.get_distance();
    Encoder_update();
    distance_judge();

    // run_flag=1;
    if(run_flag==1)
    {   //33100/2 16550         15*35  




    //     //printf("pit!\n");



       
       Dis_Out = Dis_PID_Calculate(&Dis_1,-Image_out,Dis_Speed); //- 100*(Image.Error - Image.Last_Error);//角速度环icm_data.gyro_z * K+ 400*(Image->Error - Image->Last_Error
       
 

    //    float K_turn=2;
    //     V_out=Speed_PID_Cal(&Velocity,speed_goal,MaX_Speed);

    //    if(Dis_Out>0)
    //  {
    //        PWM_R=V_out-Dis_Out*K_turn;//
    //        PWM_L=V_out+Dis_Out*(2-K_turn);//
    //  }
    //    if(Dis_Out<=0)
    //  {
    //        PWM_R=V_out-Dis_Out*(2-K_turn);//
    //        PWM_L=V_out+Dis_Out*K_turn;//
    //  }

       float K_turn=2.0;//

       if(Dis_Out>0)
     {
           v_right_target=speed_goal-Dis_Out*K_turn;//
           v_left_target=speed_goal+Dis_Out*(2-K_turn);//
     }
       if(Dis_Out<=0)
     {
           v_right_target=speed_goal-Dis_Out*(2-K_turn);//
           v_left_target=speed_goal+Dis_Out*K_turn;//S
     }

        PWM_R = Speed_PID_Cal(&Velocity_R,v_right_target,encoder_R.speed)+D_ERR*0+0* (encoder_L.D_speed - encoder_R.D_speed)+0*Dis_1.Integral;//
        PWM_L = Speed_PID_Cal(&Velocity_L,v_left_target,encoder_L.speed)-D_ERR*0-0* (encoder_L.D_speed - encoder_R.D_speed)-0*Dis_1.Integral;//
        

        set_pwm(PWM_L,PWM_R); 
        //printf("Yaw:%f\n",icm_data.yaw);
    //set_pwm(1000,1000);
    //    sprintf((char*)Tcp_buffer,"%d,%f,%f,%f\n",(encoder_L.count_now - encoder_R.count_now),Tpm_Dis,G_dis,Dis_Speed);
    //   sprintf((char*)Tcp_buffer,"%f,%.1f,%f,%f,%f,%f,%f,%f\n",speed_goal,Now_Speed,Dir_err,-Image_out,Dis_Out,Dis_Speed,Tpm_Dis,G_dis);
    //    sprintf((char*)Tcp_buffer,"%f,%f,%f,%f,%f,%f\n",Image_out,Dis_Out,v_right_target,encoder_R.speed,v_left_target,encoder_L.speed);
    //  tcp_client_dev.send_data(Tcp_buffer,strlen((char*)Tcp_buffer));
    }
    if(run_flag==2)
    {
        speed_goal=0;
        set_pwm(0,0);
        esc_pwm.set_duty(500);
    }

   it_time++;
   
    // if(it_time%20==0)
    // {
    //     printf("Yaw:%.3f   v_right_target:%.1f  Flag.picture:%d   Flag.Zebra_cross:%d  v_left_target:%.1f  imgInfo.top:%d    Flag.ramp:%d  real_distance[red_y_mid]:%.2f Dir_Err:%.1f   Huandao_R:%d    Huandao_L:%d   V_max:%.1f\n"
    //         ,icm_data.yaw,v_right_target,Flag.picture,Flag.Zebra_cross,v_left_target,imgInfo.top,Flag.ramp,real_distance[red_y_mid],Dir_err,Flag.Huandao_R,Flag.Huandao_L,MAX(encoder_L.speed,encoder_R.speed));
    //     // printf("PWM_L:%d.PWM_R:%d\n",PWM_L,-PWM_R);
    //                     //    printf("encoder_L.count_now:%d,encoder_R.count_now:%d\n",encoder_L.count_now,encoder_R.count_now);
    // }

}
    

int main(int, char**) 
{ 
    init();
    //zf_model_init();
    pid_init();
    image_init();
//    tcp_client_dev.init(SERVER_IP, PORT);
    //Flag.picture = 2;
    pit_timer.init_ms(5, pit_callback);//5ms定时器初始化

    //set_pwm(1000,1000);

    Param_Init();
    while(1)

    {

            if(Flag.picture>0||Flag.small_rock!=0||Flag.ramp!=0||Flag.Zebra_cross==1||Flag.Zebra_cross==4
                ||Flag.Huandao_L==2||Flag.Huandao_R==2 ||Flag.Huandao_L==4||Flag.Huandao_R==4 ||Flag.Huandao_L==6||Flag.Huandao_R==6
                )//
            {beep.set_level(1);}
             else{beep.set_level(0);} 

                // printf("Yaw:%f   distance_picture:%f   Flag.picture:%d   Flag.Huandao_R:%d  Flag.Huandao_L:%d\n",icm_data.yaw,distance_picture,Flag.picture,Flag.Huandao_R,Flag.Huandao_L);
            // printf("Flag.Redblock:%d   err:%d\n",FlaZXg.Redblock,abs((Right_Sideline[center.y]+Left_Sideline[center.y])-center.x));
            //  printf("center.y:%dcenter.x:%derr:%d    mid:%d\n",red_y_mid,red_x_mid,abs((Right_Sideline[red_y_mid]+Left_Sideline[red_y_mid])/2-red_x_mid),(Right_Sideline[red_y_mid]+Left_Sideline[red_y_mid])/2);
            //         printf("Flag.Redblock:%d   err_picture:%f    x_err_red:%d\n",Flag.Redblock,err_picture,x_err_red);
             frame_count++;
        // 5. 每秒统计一次帧率s
        auto current_time = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> elapsed = current_time - last_print_time;
        
        // if (elapsed.count() >= PRINT_INTERVAL)
        // {
        //     // 计算实际帧率：帧数 / 耗时（秒）
        //     double fps = frame_count / (elapsed.count() / 1000.0);
        //     // printf("top:%d,Dir_err:%.1f\n",imgInfo.top,Dir_err);
        //     //         printf("Yaw:%f,distance=%f\n",icm_data.yaw,distance);
        //            printf("encoder_L.count_now:%d,encoder_R.count_now:%d\n",encoder_L.count_now,encoder_R.count_now);
    
            
        //     // 输出帧率信息
        //     std::cout << "实时帧率：" << fps << " FPS | 累计帧数：" << frame_count << std::endl;
            
        //     // 重置统计变量
        //     frame_count = 0;
        //     last_print_time = current_time;
        // }

         
        if(run_flag==2||run_flag==0)
        {
            oled_show();
            key_scan();
        }
        // //auto start = std::chrono::high_resolution_clock::now();
         ImageDeal();
        // frame_cnt++;


        //auto end = std::chrono::high_resolution_clock::now();
        //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        //std::cout << "函数耗时: " << duration.count() / 1000.0 << " 毫秒" << std::endl;
        // 短延时10毫秒，降低CPU占用率，同时防止循环执行过快导致的资源抢占问题
        //system_delay_ms(1);

        
    }
}