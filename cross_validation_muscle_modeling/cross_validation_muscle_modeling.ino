/*
 * Program to stimulate muscle with impulse signal and record force response
 * ENS2020 Pin configuration    Arduino MEGA 256 Pins
 * 1 - GND                                GND (pin 44 unused)
 * 2 - Not connected                      45
 * 3 - SEL                                46
 * 4 - DOWN (decrement)                   47
 * 5 - BWD (Backward)                     48
 * 6 - OFF                                49
 * 7 - FWD (Forward)                      50
 * 8 - UP (increment)                     51
 * 9 - MENU                               52
 * 10 - ON                                53
 * By default pins should be HIGH
 */

#define array_len( x )  ( sizeof( x ) / sizeof( *x ) )

const int ens_ON_pin = 53;
const int ens_OFF_pin = 49;

const int ens_MENU_pin = 52;
const int ens_SEL_pin = 46;

const int ens_FWD_pin = 50;
const int ens_BWD_pin = 48;

const int ens_UP_pin = 51;
const int ens_DOWN_pin = 47;


//#include "Arduino.h"s
//#include "string.h"
//#include "HardwareSerial.h"
//#include "Wire.h"

/*
 * Pin assignment
 */
//int stimulus_onset_pin = 22;
//int movement_onset_pin = 24;
int stim_received_LED_pin = 27;
int end_of_stim_LED_pin  = 31;    // Use to indicate end of trial
int analog_force_sensor_pin = A7; // A15 is not bad -- just loose connection
int analog_stim_voltage_pin = A8;
int AIN1_pin = 5;     // Negative input pin of Analog comparator; Positive pin is internal reference = 1.1V
//int ledPin = 13;
int test_trigger_pin = 12;

/*
 * Define global variables
 */
//int trigger_duration = 53036; // trigger low for 200 ms; should be > sampling interval
//int trigger_duration_ms = 200;
//float sample_time = 0;
//int trial_num = 0;
//volatile int force_sensor = 0;
//volatile int stim_voltage = 0;
//int stimulus_onset_value = 5; // whether trigger is high(5) or low(0)
//int movement_onset_value = 5; // whether trigger is high(5) or low(0)
//const int force_threshold = 50;

// 3036 - 1Hz; //34286 - 2Hz; //53036 - 5 Hz; //64911 - 100 Hz; // 65411 - 500 Hz  // preload timer 65536-16MHz/256/2Hz
const int force_sampling_rate = 64911; // 100 Hz sampling with prescalar = 1/256
const int stim_sampling_rate = 61536; // 4 Khz sampling with no prescalar
const int num_stim_samples_reqd = 2500; // @ 4Khz sampling freq, 2500 samples = 625 ms data
const int num_force_samples_reqd = 200; // @ 100Hz sampling freq, 200 samples = 2 sec.
const int keypress_interval_very_long = 500;
const int keypress_interval_long = 200;
const int keypress_int_short = 150;
const int Number_of_trials_per_inc = 5;
int Total_num_of_stim_repetitions = 0;
volatile boolean stim_recvd = false;
volatile int stim_sample_counter = 0;
volatile int force_sample_counter = 0;
volatile int stim_voltage_array[num_stim_samples_reqd];
volatile int force_response_array[num_force_samples_reqd];
//volatile unsigned long prev_pulse_time = 0;
volatile boolean data_capture_ON = false;


void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);        // Turn on the Serial Port

  // Configure ENS - Set MENU, SEL, UP and DOWN pins as output and set them to HIGH; rest as input
  pinMode(ens_ON_pin, INPUT);
  pinMode(ens_OFF_pin, INPUT);
  pinMode(ens_FWD_pin, INPUT);
  pinMode(ens_BWD_pin, INPUT);

  pinMode(ens_DOWN_pin, OUTPUT);
  digitalWrite(ens_DOWN_pin, HIGH);
  pinMode(ens_UP_pin, OUTPUT);
  digitalWrite(ens_UP_pin, HIGH);
  pinMode(ens_MENU_pin, OUTPUT);
  digitalWrite(ens_MENU_pin, HIGH);
  pinMode(ens_SEL_pin, OUTPUT);
  digitalWrite(ens_SEL_pin, HIGH);

  // Configure Arduino LED, ADC and Analog Comparator pins
  /////Inputs
  pinMode(analog_force_sensor_pin, INPUT);                  // Force sensor
  pinMode(analog_stim_voltage_pin, INPUT);                  // Electrical stimulation voltage sensor
  pinMode(AIN1_pin, INPUT);

  ////Outputs
  pinMode(stim_received_LED_pin, OUTPUT);
  digitalWrite(stim_received_LED_pin, LOW);
  pinMode(end_of_stim_LED_pin, OUTPUT);
  digitalWrite(end_of_stim_LED_pin, LOW);
  pinMode(test_trigger_pin, OUTPUT);
  digitalWrite(test_trigger_pin, LOW);

  //Configure Interrupts - Timers, Analog Comparators
  noInterrupts();     // disable all software interrupts
  ACSR = B01010010;   // Analog comparator control and status register
  /*ACSR =
    (0<<ACD) | // Analog Comparator Disabled
    (1<<ACBG) | // Analog Comparator Bandgap Select: AIN0 is connected to 1.1V internal reference
    (0<<ACO) | // Analog Comparator Output: Off
    (1<<ACI) | // Analog Comparator Interrupt Flag: Clear Pending Interrupt
    (0<<ACIE) | // Analog Comparator Interrupt: Enabled
    (0<<ACIC) | // Analog Comparator Input Capture: Disabled
    (1<<ACIS1) | (0<ACIS0); // Analog Comparator Interrupt Mode: Comparator Interrupt on Falling Output Edge of comparator, by default comparator output should be high
  */

  // Timer 1 for reading values from force sensor
  TCCR1A = 0x00;
  TCCR1B = 0x00;            // Timer stopped, to start timer - load 1/256 prescalar = 0x04
  TCNT1 =  force_sampling_rate;
  TIMSK1 |= (1 << TOIE1);   // enable timer overflow interrupt

  // Timer 3 controls the stimulus-onset trigger
  TCCR3A = 0x00;
  TCCR3B = 0x00;            // Timer stopped, to start timer - with prescalar = 0x01
  TCNT3 =  stim_sampling_rate;
  TIMSK3 |= (1 << TOIE3);   // enable timer overflow interrupt
  interrupts();             // enable all software interrupts

  //Get user/python input for to start experiment
  Serial.print("Enter number of model validation stimulus trains (1-10). Waiting for Input....");
  while (Serial.available() == 0);
  Total_num_of_stim_repetitions = Serial.parseInt();  // Only accepts integers
  Serial.println(Total_num_of_stim_repetitions);
  //ACSR |= (1 << ACIE);      // Now enable analog comaprator interrupt
}

void loop() {
  // put your main code here, to run repeatedly:
  for (int rep_num = 1; rep_num <= Total_num_of_stim_repetitions; rep_num++) {
    delay(3000); // Wait 3 sec
    //Serial.print("Enter loop: ");
    //Serial.println(rep_num);

    //Turn ON the stimulator
    digitalWrite(ens_SEL_pin, LOW);
    delay(keypress_interval_very_long);
    digitalWrite(ens_SEL_pin, HIGH);
    delay(keypress_interval_very_long);

    // Increase by 10 increments immediately
    for (int i = 1; i <= 10; i++) {
      digitalWrite(ens_UP_pin, LOW);
      delay(keypress_int_short);
      digitalWrite(ens_UP_pin, HIGH);
      delay(keypress_int_short);
    }
    ACSR |= (1 << ACI); // Clear pending interrupts
    ACSR |= (1 << ACIE);      // Now enable analog comaprator interrupt
    // Wait for 5 trials
    for (int trial_no = 1; trial_no <= Number_of_trials_per_inc; trial_no++) {
      //Serial.println("Inc 1, Trial " + String(trial_no));
      while (!stim_recvd) {     // 1st measurement
        digitalWrite(end_of_stim_LED_pin, HIGH);
        delay(100);
        digitalWrite(end_of_stim_LED_pin, LOW);
        delay(100);
      }
      stim_recvd = false;
    }

    // Increase again by 10 increments immediately
    for (int i = 1; i <= 10; i++) {
      digitalWrite(ens_UP_pin, LOW);
      delay(keypress_int_short);
      digitalWrite(ens_UP_pin, HIGH);
      delay(keypress_int_short);
    }
    // Wait for 5 trials
    for (int trial_no = 1; trial_no <= Number_of_trials_per_inc; trial_no++) {
      //Serial.println("Inc 2, Trial " + String(trial_no));
      while (!stim_recvd) {     // 1st measurement
        digitalWrite(end_of_stim_LED_pin, HIGH);
        delay(100);
        digitalWrite(end_of_stim_LED_pin, LOW);
        delay(100);
      }
      stim_recvd = false;
    }

    // Once again, increase by 5 increments immediately
    for (int i = 1; i <= 5; i++) {
      digitalWrite(ens_UP_pin, LOW);
      delay(keypress_int_short);
      digitalWrite(ens_UP_pin, HIGH);
      delay(keypress_int_short);
    }
    // Wait for 5 trials
    for (int trial_no = 1; trial_no <= Number_of_trials_per_inc; trial_no++) {
      //Serial.println("Inc 3, Trial " + String(trial_no));
      while (!stim_recvd) {     // 1st measurement
        digitalWrite(end_of_stim_LED_pin, HIGH);
        delay(100);
        digitalWrite(end_of_stim_LED_pin, LOW);
        delay(100);
      }
      stim_recvd = false;
    }

    // Finally, increase by 5 increments immediately
    for (int i = 1; i <= 5; i++) {
      digitalWrite(ens_UP_pin, LOW);
      delay(keypress_int_short);
      digitalWrite(ens_UP_pin, HIGH);
      delay(keypress_int_short);
    }
    // Wait for 5 trials
    for (int trial_no = 1; trial_no <= Number_of_trials_per_inc; trial_no++) {
      //Serial.println("Inc 3, Trial " + String(trial_no));
      while (!stim_recvd) {     // 1st measurement
        digitalWrite(end_of_stim_LED_pin, HIGH);
        delay(100);
        digitalWrite(end_of_stim_LED_pin, LOW);
        delay(100);
      }
      stim_recvd = false;
    }

    // Turn OFF the stimulator
    digitalWrite(ens_MENU_pin, LOW);
    delay(keypress_interval_very_long);
    digitalWrite(ens_MENU_pin, HIGH);

    delay(1000);
    Serial.println("");
    Serial.println("Rep " + String(rep_num) + "of " + String(Total_num_of_stim_repetitions) + " completed. Continue? [y/n]: ");
    while (Serial.read() != 'y') {
      digitalWrite(end_of_stim_LED_pin, HIGH);
      delay(1000);
      digitalWrite(end_of_stim_LED_pin, LOW);
      delay(1000);
    }
    /*if (Serial.read() == 'n'){
      Serial.println("Exiting program");
      break;
    }*/

  }
  digitalWrite(end_of_stim_LED_pin, HIGH);
  //Serial.println("q"); // Exit python program
  //Do not disable all interrupts
  while (1);
}

ISR(ANALOG_COMP_vect) {
  ACSR |= (0 << ACIE) | (1 << ACI); // Disable analog comparator interrupt until this pin is captured
  stim_voltage_array[stim_sample_counter] = analogRead(analog_stim_voltage_pin);
  force_response_array[force_sample_counter] = analogRead(analog_force_sensor_pin);
  stim_sample_counter++;
  force_sample_counter++;

  // Measure interpulse interval
  //unsigned long new_pulse_time = millis();
  //Serial.print("IPI = ");
  //Serial.println(new_pulse_time - prev_pulse_time);
  //prev_pulse_time = new_pulse_time;

  if (data_capture_ON == false) {
    // If data is not already being sampled - to avoid over-writing the arrays
    // Start data sampling timers
    TCCR3B = 0x01; // measure stim voltage using TIMER 3
    TCCR1B = 0x04; // measure force response using TIMER 1
    digitalWrite(stim_received_LED_pin, HIGH);
    data_capture_ON == true;
  }
}

ISR(TIMER1_OVF_vect) {
  TCNT1 = force_sampling_rate;
  force_response_array[force_sample_counter] = analogRead(analog_force_sensor_pin);
  force_sample_counter++;

  if (force_sample_counter >= array_len(force_response_array)) {
    force_sample_counter = 0;
    // Stop timer and send data to computer over serial port
    TCCR1B = 0x00;
    data_capture_ON = false; // Program stopped data capturing
    //Serial.println("Sending force measurements");
    for (int cnt = 0; cnt < array_len(force_response_array); cnt++) {
      Serial.print(force_response_array[cnt]);
      if (cnt < array_len(force_response_array) - 1) {
        Serial.print(",");
      }
    }
    Serial.println("");
    Serial.flush();   // Wait until all data has been sent
    //Serial.println("Force measurements sent");
  }
}

ISR(TIMER3_OVF_vect) {
  TCNT3 = stim_sampling_rate;
  stim_voltage_array[stim_sample_counter] = analogRead(analog_stim_voltage_pin);
  stim_sample_counter++;

  if (stim_sample_counter >= array_len(stim_voltage_array)) {
    // Stop timer and send data to computer over serial port
    TCCR3B = 0x00;
    stim_sample_counter = 0;
    ACSR |= (1 << ACI);  // One trial complete- Now enable analog comparator interrupt and clear pending interrupts
    ACSR |= (1 << ACIE); // Need to ensure that sencond stimulus doesn't appear with 2 seconds of first stimulus

    // Tell main loop that stim is received so it can increment stimulator
    stim_recvd = true;
    digitalWrite(stim_received_LED_pin, LOW);

    for (int cnt = 0; cnt < array_len(stim_voltage_array); cnt++) {
      Serial.print(stim_voltage_array[cnt]);
      if (cnt < array_len(stim_voltage_array) - 1) {
        Serial.print(",");
      }
    }
    Serial.println("");
    Serial.flush();   // Wait until all data has been sent
  }
}
