use apus_core::ApusError;
use parking_lot::Mutex;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermalZone {
    /// < 65°C — Green Zone (full-speed operation)
    GreenOptimal,
    /// 65..69°C — Yellow Zone (PID micro-sleep pacing to cool the NVMe controller)
    YellowPacing,
    /// >= 70°C — Red Zone (emergency thermal brake to protect system NVMe from hardware throttling)
    RedEmergencyBrake,
}

/// S.M.A.R.T. temperature monitor and PID thermal controller for NVMe SSDs
pub struct SmartThermalController {
    target_temp_c: u8,
    yellow_zone_c: u8,
    red_zone_c: u8,
    current_temp_c: Mutex<u8>,
    integral_error: Mutex<f32>,
    last_error: Mutex<f32>,
}

impl SmartThermalController {
    pub fn new(target_temp_c: u8, yellow_zone_c: u8, red_zone_c: u8) -> Self {
        Self {
            target_temp_c,
            yellow_zone_c,
            red_zone_c,
            current_temp_c: Mutex::new(52),
            integral_error: Mutex::new(0.0),
            last_error: Mutex::new(0.0),
        }
    }

    pub fn update_sensor_temp(&self, temp_c: u8) {
        *self.current_temp_c.lock() = temp_c;
    }

    pub fn current_temp_c(&self) -> u8 {
        *self.current_temp_c.lock()
    }

    pub fn zone(&self) -> ThermalZone {
        let t = self.current_temp_c();
        if t >= self.red_zone_c {
            ThermalZone::RedEmergencyBrake
        } else if t >= self.yellow_zone_c {
            ThermalZone::YellowPacing
        } else {
            ThermalZone::GreenOptimal
        }
    }

    /// Evaluates the thermal zone and computes a micro-cooldown delay using a PID controller.
    /// In the Red Zone (>= 70°C), returns `ApusError::ThermalThrottleTripped`.
    pub fn check_and_compute_pid_cooldown(&self) -> Result<Duration, ApusError> {
        let temp = self.current_temp_c();
        if temp >= self.red_zone_c {
            return Err(ApusError::ThermalThrottleTripped { temp_c: temp });
        }
        if temp < self.target_temp_c {
            *self.integral_error.lock() = 0.0;
            *self.last_error.lock() = 0.0;
            return Ok(Duration::ZERO);
        }

        // PID controller to smoothly pace IOPS as temperature approaches the yellow zone
        let error = (temp as f32) - (self.target_temp_c as f32);
        let mut integral = self.integral_error.lock();
        let mut prev_err = self.last_error.lock();

        *integral = (*integral + error).clamp(0.0, 50.0);
        let derivative = error - *prev_err;
        *prev_err = error;

        let kp = 180.0;
        let ki = 25.0;
        let kd = 40.0;
        let delay_micros = (kp * error + ki * (*integral) + kd * derivative).max(0.0) as u64;

        Ok(Duration::from_micros(delay_micros.min(15_000)))
    }
}
