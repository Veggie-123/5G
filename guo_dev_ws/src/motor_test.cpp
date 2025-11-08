// Motor control test using pigpio 
#include <iostream>
#include <pigpio.h>
#include <unistd.h>
#include <cstdlib>

// 定义舵机和电机引脚号、PWM范围、PWM频率、PWM占空比解锁值
const int servo_pin = 12; // 存储舵机引脚号
const float servo_pwm_range = 10000.0; // 存储舵机PWM范围
const float servo_pwm_frequency = 50.0; // 存储舵机PWM频率
const float servo_pwm_duty_cycle_unlock = 730.0; // 存储舵机PWM占空比解锁值

const int motor_pin = 13; // 存储电机引脚号
// const float motor_pwm_range = 36300.0; // 存储电机PWM范围
const float motor_pwm_range = 40000.0; // 存储电机PWM范围
const float motor_pwm_frequency = 200.0; // 存储电机PWM频率
const float motor_pwm_duty_cycle_unlock = 11400.0; // 存储电机PWM占空比解锁值

//---------------------------------------------------------------------------------------------------
float motor_pwm_mid = motor_pwm_duty_cycle_unlock; // 存储舵机中值
//---------------------------------------------------------------------------------------------------

const int yuntai_LR_pin = 22; // 存储云台引脚号
const float yuntai_LR_pwm_range = 1000.0; // 存储云台PWM范围
const float yuntai_LR_pwm_frequency = 50.0; // 存储云台PWM频率
const float yuntai_LR_pwm_duty_cycle_unlock = 63.0; //大左小右 


const int yuntai_UD_pin = 23; // 存储云台引脚号
const float yuntai_UD_pwm_range = 1000.0; // 存储云台PWM范围
const float yuntai_UD_pwm_frequency = 50.0; // 存储云台PWM频率
const float yuntai_UD_pwm_duty_cycle_unlock = 58.0; //大上下小

int parkchose = 2; // 停车车库检测结果


// 定义舵机和电机PWM初始化函数
void servo_motor_pwmInit(void) 
{
    if (gpioInitialise() < 0) // 初始化GPIO，如果失败则返回
    {
        std::cout << "GPIO failed ! Please use sudo !" << std::endl; // 输出失败信息
        return; // 返回
    }
    else
        std::cout << "GPIO ok. Good !!" << std::endl; // 输出成功信息

    gpioSetMode(servo_pin, PI_OUTPUT); // 设置舵机引脚为输出模式
    gpioSetPWMfrequency(servo_pin, servo_pwm_frequency); // 设置舵机PWM频率
    gpioSetPWMrange(servo_pin, servo_pwm_range); // 设置舵机PWM范围
    gpioPWM(servo_pin, servo_pwm_duty_cycle_unlock); // 设置舵机PWM占空比解锁值

    gpioSetMode(motor_pin, PI_OUTPUT); // 设置电机引脚为输出模式
    gpioSetPWMfrequency(motor_pin, motor_pwm_frequency); // 设置电机PWM频率
    gpioSetPWMrange(motor_pin, motor_pwm_range); // 设置电机PWM范围
    gpioPWM(motor_pin, motor_pwm_duty_cycle_unlock); // 设置电机PWM占空比解锁

    gpioSetMode(yuntai_LR_pin, PI_OUTPUT); // 设置云台引脚为输出模式
    gpioSetPWMfrequency(yuntai_LR_pin, yuntai_LR_pwm_frequency); // 设置云台PWM频率
    gpioSetPWMrange(yuntai_LR_pin, yuntai_LR_pwm_range); // 设置云台PWM范围
    gpioPWM(yuntai_LR_pin, yuntai_LR_pwm_duty_cycle_unlock); // 设置云台PWM占空比解锁值

    gpioSetMode(yuntai_UD_pin, PI_OUTPUT); // 设置云台引脚为输出模式
    gpioSetPWMfrequency(yuntai_UD_pin, yuntai_UD_pwm_frequency); // 设置云台PWM频率
    gpioSetPWMrange(yuntai_UD_pin, yuntai_UD_pwm_range); // 设置云台PWM范围
    gpioPWM(yuntai_UD_pin, yuntai_UD_pwm_duty_cycle_unlock); // 设置云台PWM占空比解锁值
    
}

// 定义舵机和电机PWM控制函数
void servo_motor_pwmControl(float servo_pwm, float motor_pwm)
{
    gpioPWM(servo_pin, servo_pwm); // 控制舵机PWM
    gpioPWM(motor_pin, motor_pwm); // 控制电机PWM
}

void gohead(){
    if(parkchose == 1 ){ //try to find park A
        gpioPWM(13, motor_pwm_mid + 2800);
        gpioPWM(13, motor_pwm_mid + 800); // 设置电机PWM
        gpioPWM(12, 690); // 设置舵机PW0M
        sleep(1);
        gpioPWM(13, motor_pwm_mid + 800); // 设置电机PWM
        gpioPWM(12, 780); // 设置舵机PW0M
        sleep(2);
        std::cout << "gohead--------------------------------------------------------------Try To Find Park AAAAAAAAAAAAAAA" << std::endl;
    }
    else if(parkchose == 2){ //try to find park B
        gpioPWM(13, motor_pwm_mid + 2800);
        gpioPWM(13, motor_pwm_mid + 800); // 设置电机PWM
        gpioPWM(12, 690); // 设置舵机PWM
        sleep(1);
        gpioPWM(13, motor_pwm_mid + 800); // 设置电机PWM
        gpioPWM(12, 640); // 设置舵机PW0M
        sleep(2);
        gpioPWM(13, motor_pwm_mid);
    }
}

// 主函数
int main(void)
{   
    gpioTerminate(); // 终止GPIO
    servo_motor_pwmInit(); // 舵机和电机PWM初始化

    float servo_pwm1 = 800; // 定义舵机PWM
    float motor_pwm2 = 11000; // 定义电机PWM

    // servo_motor_pwmControl(servo_pwm1, motor_pwm2); // 控制舵机和电机PWM

    // sleep(3); // 等待3秒

    // servo_motor_pwmControl(0, 0); // 停止舵机和电机PWM

    // gpioPWM(12, 815); // 设置舵机PWM
    // gpioPWM(13, 11000); // 设置电机PWM

    // sleep(2); // 增加1秒延迟，确保舵机停止

    //----------------------------------向右变道--------------
    // gpioPWM(12, 700); // 设置舵机PWM
    // gpioPWM(13, 12000); // 设置电机PWM
    // usleep(900000);
    // gpioPWM(12, 870); // 设置舵机PWM
    // gpioPWM(13, 11500); // 设置电机PWM
    // usleep(900000); // 延时550毫秒
    // gpioPWM(12, 815); // 设置舵机PWM
    // gpioPWM(13, 11500); // 设置电机PWM
    
    // gpioPWM(12, servo_pwm_mid); // 设置舵机PWM
    // gpioPWM(13, 11300); // 设置电机PWM

    gpioPWM(13,motor_pwm_mid + 00); // 设置电机PWM
    
    // sleep(3);

    gpioPWM(12, servo_pwm_duty_cycle_unlock); // 设置舵机PWM
    gpioPWM(13, motor_pwm_mid + 00); // 设置电机PWM
    sleep(10); 


    // gpioPWM(13,motor_pwm_duty_cycle_unlock + 000); // 设置电机PWM

    //system("sudo -u pi /home/pi/.nvm/versions/node/v12.22.12/bin/node /home/pi/network-rc/we2hdu.js"); // 播放音频文件

    //----------------------------------向左变道--------------
    // gpioPWM(12, 800); // 设置舵机PWM
    // gpioPWM(13, 12500); // 设置电机PWM
    // usleep(1500000);
    // gpioPWM(12, 630); // 设置舵机PWM
    // gpioPWM(13, 12500); // 设置电机PWM
    // usleep(1500000); // 延时550毫秒
    // gpioPWM(12, servo_pwm_mid); // 设置舵机PWM
    // gpioPWM(13, motor_pwm_duty_cycle_unlock); // 设置电机PWM
    // sleep(1);

    // for(int i = 0 ; i < 10 ; i++){
        // gpioPWM(12, 620); // 设置舵机PWM
        // gpioPWM(13, 11300); // 设置电机PWM  
        // usleep(1400000);
        // gpioPWM(12, 820); // 设置舵机PWM
        // gpioPWM(13, 11300); // 设置电机PWM
        // usleep(500000); // 延时550毫秒
        // gpioPWM(12, servo_pwm_mid); // 设置舵机PWM
        // gpioPWM(13, 11400); // 设置电机PWM
    // }
    //----------------------------------向左变道--------------
  
    // gpioPWM(12, servo_pwm_mid); // 设置舵机PWMs
    // gpioPWM(13, 10000); // 设置电机PWM

    return 0; // 返回
}