//#include "Arduino.h"s
//#include "string.h"
//#include "HardwareSerial.h"
//#include "Wire.h"

// Pin Mappings EEG triggers <---> Arduino digital pins
//                  S1       <--->      34
//                  S2       <--->      32
//                  S4       <--->      30
//                  S8       <--->      28
//                  S16      <--->      26
//                  S32      <--->      NC
//                  S64      <--->      NC
//                  S128     <--->      NC

int EEG_S1_trigger = 34;
int EEG_S2_trigger = 32; //previously called, stimulus_onset_pin = 32;
int EEG_S4_trigger = 30;
int EEG_S8_trigger = 28;
int EEG_S16_trigger = 26; // previously called, movement_onset_pin = 26; 


int target_LED_pin = 27;    
int error_LED_pin  = 31;    // Use to indicate false start 
int analog_force_sensor_pin = A8; 
int analog_stim_voltage_pin = A15; 
int Total_num_of_trials = 0;
int ledPin = 13;

// 3036 - 1Hz; //34286 - 2Hz; //53036 - 5 Hz; //64911 - 100 Hz; // 654911 - 500 Hz  // preload timer 65536-16MHz/256/2Hz
//int sampling_rate = 64911; // No longer used
int ctc_mode_sampler_value = 2499; //  100 Hz timer1, ctc mode, prescalar = 1/64; OCR1A = (16Mhz/prescalar(64)/100Hz) - 1
int trigger_duration = 53036; // trigger low for 200 ms; should be > sampling interval
//int trigger_duration_ms = 200;

volatile float sample_time = 0;
int trial_num = 0;
volatile int force_sensor = 0;
volatile int stim_voltage = 0;
int stimulus_onset_value = 5; // whether trigger is high(5) or low(0) 
int movement_onset_value = 5; // whether trigger is high(5) or low(0) 
int target_reached_value = 5; // whether trigger is high(5) or low(0) 
const int force_threshold = 70;

/* 
 *  ISR for Timer 1 - Sends serial data for logging to Python GUI
 */
/* ISR(TIMER1_OVF_vect)        // interrupt service routine that wraps a user defined function supplied by attachInterrupt
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
}*/
ISR(TIMER1_COMPA_vect){
  digitalWrite(ledPin, digitalRead(ledPin) ^ 1);
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
  Serial.print(movement_onset_value);
  Serial.print(" ");
  Serial.println(target_reached_value);
}

ISR(TIMER3_OVF_vect){
  TCNT3 =  trigger_duration;
  TCCR3B = 0x00;    // Stop Timer
  digitalWrite(EEG_S2_trigger,HIGH);
  stimulus_onset_value = 5;
  digitalWrite(EEG_S16_trigger,HIGH);
  movement_onset_value = 5;
  digitalWrite(EEG_S8_trigger,HIGH);
  target_reached_value = 5;
}

/*ISR(TIMER4_OVF_vect){
  TCNT4 =  trigger_duration;
  TCCR4B = 0x00;    // Stop Timer
  digitalWrite(EEG_S16_trigger,HIGH);
  movement_onset_value = 5;
}*/


void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);        // Turn on the Serial Port

  // if analog input pin 0 is unconnected, random analog
  // noise will cause the call to randomSeed() to generate
  // different seed numbers each time the sketch runs.
  // randomSeed() will then shuffle the random function.
  randomSeed(analogRead(0));
  

  pinMode(EEG_S1_trigger, OUTPUT);  
  digitalWrite(EEG_S1_trigger, HIGH);
  pinMode(EEG_S2_trigger, OUTPUT);  // Stimulus onset trigger
  digitalWrite(EEG_S2_trigger, HIGH);
  pinMode(EEG_S4_trigger, OUTPUT);  
  digitalWrite(EEG_S4_trigger, HIGH);
  pinMode(EEG_S8_trigger, OUTPUT);    // Target reach trigger
  digitalWrite(EEG_S8_trigger, HIGH);
  pinMode(EEG_S16_trigger, OUTPUT);   // Movement onset trigger
  digitalWrite(EEG_S16_trigger, HIGH);

  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, LOW);
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
  Serial.println("time,trial_num,force,stim_voltage,stimulus_onset,movment_onset,grasp_released");   
     
  noInterrupts();           // disable all software interrupts
  // Timer 1 for reading values from all pins
  //TCCR1A = 0;
  //TCCR1B = 0x04;            // Start timer - load prescalar
  //TCNT1 =  sampling_rate;    
  //TIMSK1 |= (1 << TOIE1);   // enable timer overflow interrupt

  // Timer 1 in CTC mode
  TCCR1A = 0x00;
  TCCR1B |= (1 << WGM12)|(1 << CS11)|(1 << CS10);
  TCNT1  = 0; // Initialize counter
  OCR1A = ctc_mode_sampler_value;
  TIMSK1 |= (1 << OCIE1A);   // Enable ctc interrupt

  // Timer 3 controls the stimulus-onset trigger
  TCCR3A = 0;
  TCCR3B = 0x00;            // Timer stopped
  TCNT3 =  trigger_duration;    
  TIMSK3 |= (1 << TOIE3);   // enable timer overflow interrupt


  // Timer 4 controls movement-onset trigger
  //TCCR4A = 0;
  //TCCR4B = 0x00;            // Timer stopped
  //TCNT4 =  trigger_duration;    
  //TIMSK4 |= (1 << TOIE4);   // enable timer overflow interrupt
 
  interrupts();             // enable all software interrupts 
 
  delay(500);
  // Send initialization trigger to EEG
  digitalWrite(EEG_S2_trigger,LOW);
  digitalWrite(EEG_S16_trigger,LOW);
  stimulus_onset_value = 0;
  movement_onset_value = 0;
  TCCR3B = 0x04;                  // Start Timer3 
  TCCR4B = 0x04;                  // Start Timer4
  
}

void loop() {
  float previous_sample_time = 0;
  
  // put your main code here, to run repeatedly:
  for (trial_num=1; trial_num<= Total_num_of_trials; trial_num++){
    delay(random(4000,6001));     // Fixation - random seconds
    
    // Stimulus - Send stimulus_onset trigger; Turn LED ON; Start monitoring force
    digitalWrite(EEG_S2_trigger,LOW);
    stimulus_onset_value = 0;
    digitalWrite(target_LED_pin,HIGH);
    TCCR3B = 0x04;      
    previous_sample_time = sample_time;
        
    // Movement detected - Turn LED OFF; Send movement_onset trigger;
    //delay(3000);        // movement detected
    while(force_sensor < force_threshold);
    digitalWrite(target_LED_pin,LOW);  // Turn OFF green led, irrespective of whether trial is aborted or not
    
    if ((sample_time - previous_sample_time) > 2){
      digitalWrite(EEG_S16_trigger,LOW);
      movement_onset_value = 0;
      //TCCR4B = 0x04;
      TCCR3B = 0x04;
          
      while(force_sensor >= force_threshold){       // wait for subject to relax    
        digitalWrite(target_LED_pin,HIGH);
        delay(200);
        digitalWrite(target_LED_pin,LOW);      
        delay(200);
      }
      digitalWrite(EEG_S8_trigger,LOW);
      target_reached_value = 0;
      TCCR3B = 0x04;
    }
    else{
      // Abort trial, decrement trial_num by 1, 
      trial_num--;
      // Don't generate any triggers
      // Blink error led until sensor is released

      while(force_sensor >= force_threshold){       // wait for subject to relax    
        digitalWrite(error_LED_pin,HIGH);
        delay(200);
        digitalWrite(error_LED_pin,LOW);      
        delay(200);
      }
    }
    
    
  }

  // Send end of block trigger to EEG
  delay(3000); 
  digitalWrite(EEG_S2_trigger,LOW);
  digitalWrite(EEG_S16_trigger,LOW);
  stimulus_onset_value = 0;
  movement_onset_value = 0;
  TCCR3B = 0x04;                  // Start Timer3 
  TCCR4B = 0x04;                  // Start Timer4    
  TCCR1B = 0;                     // Stop Timer for data logging
  digitalWrite(error_LED_pin,HIGH);             // Keep red LED ON to indicate end of trial
  while(1);                       // Wait forever 
}
