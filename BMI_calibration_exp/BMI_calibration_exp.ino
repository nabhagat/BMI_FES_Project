//#include "Arduino.h"s
//#include "string.h"
//#include "HardwareSerial.h"
//#include "Wire.h"

int stimulus_onset_pin = 26;
int movement_onset_pin = 32; 
int target_LED_pin = 27;    
int error_LED_pin  = 31;    // Use to indicate false start 
int analog_force_sensor_pin = A15; 
int analog_stim_voltage_pin = A14; 
int Total_num_of_trials = 0;
int ledPin = 13;

// 3036 - 1Hz; //34286 - 2Hz; //53036 - 5 Hz; //64911 - 100 Hz; // 654911 - 500 Hz  // preload timer 65536-16MHz/256/2Hz
int sampling_rate = 34286; //64911;  
int trigger_duration = 53036; // trigger low for 200 ms; should be > sampling interval
//int trigger_duration_ms = 200;

float sample_time = 0;
int trial_num = 0;
volatile int force_sensor = 0;
volatile int stim_voltage = 0;
int stimulus_onset_value = 5; // whether trigger is high(5) or low(0) 
int movement_onset_value = 5; // whether trigger is high(5) or low(0) 
const int force_threshold = 50;

/* 
 *  ISR for Timer 1 - Sends serial data for logging to Python GUI
 */
 ISR(TIMER1_OVF_vect)        // interrupt service routine that wraps a user defined function supplied by attachInterrupt
{
  TCNT1 = sampling_rate;            // preload timer
  //digitalWrite(ledPin, digitalRead(ledPin) ^ 1);

  sample_time += 0.01;
  force_sensor = analogRead(analog_force_sensor_pin);
  stim_voltage = analogRead(analog_stim_voltage_pin);
  Serial.print(sample_time);
  Serial.print(" ");
  Serial.print(trial_num);
  Serial.print(" ");
  Serial.print(force_sensor);
  Serial.print(" ");
  Serial.print(stim_voltage);
  Serial.print(" ");
  Serial.print(stimulus_onset_value);
  Serial.print(" ");
  Serial.println(movement_onset_value);
  //Serial.println(",");
  
  //Serial.println(digitalRead(13));
  /*Serial.println(analogRead(15));
  if (No_of_samples > 2000){
    TCCR1B = 0;     // Stop Timer 
    No_of_samples = 0;
  }
  else {
     No_of_samples+=1; 
  }
  */
}

ISR(TIMER3_OVF_vect){
  TCNT3 =  trigger_duration;
  TCCR3B = 0x00;    // Stop Timer
  digitalWrite(stimulus_onset_pin,HIGH);
  stimulus_onset_value = 5;
}

ISR(TIMER4_OVF_vect){
  TCNT4 =  trigger_duration;
  TCCR4B = 0x00;    // Stop Timer
  digitalWrite(movement_onset_pin,HIGH);
  movement_onset_value = 5;
}


void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);        // Turn on the Serial Port

  // if analog input pin 0 is unconnected, random analog
  // noise will cause the call to randomSeed() to generate
  // different seed numbers each time the sketch runs.
  // randomSeed() will then shuffle the random function.
  randomSeed(analogRead(0));
  
  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, LOW);
  pinMode(stimulus_onset_pin, OUTPUT);  // Tell Arduino that stimulus_onset_pin is an output pin
  digitalWrite(stimulus_onset_pin, HIGH);
  pinMode(movement_onset_pin, OUTPUT);  //Tell Arduino that movement_onset_pin is an output pin
  digitalWrite(movement_onset_pin, HIGH);
  pinMode(target_LED_pin, OUTPUT);  
  digitalWrite(target_LED_pin, LOW);
  pinMode(error_LED_pin, OUTPUT);  
  digitalWrite(error_LED_pin, LOW);
  pinMode(analog_force_sensor_pin, INPUT);                  // Force sensor
  pinMode(analog_stim_voltage_pin, INPUT);                  // Electrical stimulation voltage sensor

  //Get user/python input for to start experiment
  Serial.print("Enter number of trials. Waiting for Input....");
  while(Serial.available()==0);
  Total_num_of_trials = Serial.parseInt();  // Only accepts integers
  Serial.println(Total_num_of_trials);
    
  Serial.println("Force and trigger data from Arduino. Sampling rate = 100 Hz");
  Serial.println(" ");
  Serial.println("time,trial_num,force,stim_voltage,stimulus_onset,movment_onset");   
     
  noInterrupts();           // disable all software interrupts
  // Timer 1 for reading values from all pins
  TCCR1A = 0;
  TCCR1B = 0x04;            // Start timer - load prescalar
  TCNT1 =  sampling_rate;    
  TIMSK1 |= (1 << TOIE1);   // enable timer overflow interrupt

  // Timer 3 controls the stimulus-onset trigger
  TCCR3A = 0;
  TCCR3B = 0x00;            // Timer stopped
  TCNT3 =  trigger_duration;    
  TIMSK3 |= (1 << TOIE3);   // enable timer overflow interrupt


  // Timer 4 controls movement-onset trigger
  TCCR4A = 0;
  TCCR4B = 0x00;            // Timer stopped
  TCNT4 =  trigger_duration;    
  TIMSK4 |= (1 << TOIE4);   // enable timer overflow interrupt
 
  interrupts();             // enable all software interrupts 
 
  delay(500);
  // Send initialization trigger to EEG
  digitalWrite(stimulus_onset_pin,LOW);
  digitalWrite(movement_onset_pin,LOW);
  stimulus_onset_value = 0;
  movement_onset_value = 0;
  TCCR3B = 0x04;                  // Start Timer3 
  TCCR4B = 0x04;                  // Start Timer4
  
      //delay(trigger_duration_ms);
      //digitalWrite(stimulus_onset_pin,HIGH);
      //digitalWrite(movement_onset_pin,HIGH);
      //stimulus_onset_value = 5;
      //movement_onset_value = 5;
}

void loop() {
  // put your main code here, to run repeatedly:
  for (trial_num=1; trial_num<= Total_num_of_trials; trial_num++){
    delay(random(4000,6001));     // Fixation - random seconds
    
    // Stimulus - Send stimulus_onset trigger; Turn LED ON; Start monitoring force
    digitalWrite(stimulus_onset_pin,LOW);
    stimulus_onset_value = 0;
    digitalWrite(target_LED_pin,HIGH);
    TCCR3B = 0x04;      
        
    // Movement detected - Turn LED OFF; Send movement_onset trigger;
    //delay(3000);        // movement detected
    while(force_sensor < force_threshold);
    digitalWrite(movement_onset_pin,LOW);
    movement_onset_value = 0;
    digitalWrite(target_LED_pin,LOW);  
    TCCR4B = 0x04;
    digitalWrite(error_LED_pin,HIGH);
    while(force_sensor >= force_threshold);       // wait for subject to relax
    digitalWrite(error_LED_pin,LOW);      
  }

  // Send end of block trigger to EEG
  delay(3000); 
  digitalWrite(stimulus_onset_pin,LOW);
  digitalWrite(movement_onset_pin,LOW);
  stimulus_onset_value = 0;
  movement_onset_value = 0;
  TCCR3B = 0x04;                  // Start Timer3 
  TCCR4B = 0x04;                  // Start Timer4    
  TCCR1B = 0;                     // Stop Timer for data logging
  while(1);                       // Wait forever 
}
