import numpy as np
from scipy.fft import fft

class AdaptiveSovereignEngine:
    def __init__(self, frequency_lanes=128, history_horizon=10):
        """
        Initializes the self-evolving signal mapping engine.
        Tracks an adaptive baseline history matrix to adjust validation thresholds dynamically.
        """
        self.lanes = frequency_lanes
        self.horizon = history_horizon
        
        # Initialize an empty, evolving state memory matrix
        self.signature_history = []
        self.adaptive_threshold = 0.05 # Initial hyper-strict variance barrier

    def ingest_and_transform_signal(self, raw_audio_frames):
        """
        [COMPUTE] Maps spatial acoustic energy directly into a normalized frequency vector 
        using a high-performance Fast Fourier Transform (FFT).
        """
        signal_array = np.array(raw_audio_frames, dtype=np.float32)
        if len(signal_array) < self.lanes:
            # Zero-pad trailing arrays to maintain absolute matrix dimensionality
            signal_array = np.pad(signal_array, (0, self.lanes - len(signal_array)), 'constant')
            
        # Execute vectorized 1D Fourier Transform to isolate wave components
        fft_complex = fft(signal_array[:self.lanes])
        magnitude_spectrum = np.abs(fft_complex)
        
        # Normalize the structural signature to unit length to defend against volume shifts
        norm = np.linalg.norm(magnitude_spectrum)
        return magnitude_spectrum if norm == 0 else magnitude_spectrum / norm

    def register_initial_motif(self, baseline_audio_samples):
        """
        [LAW] Establishes the core structural signature mapping.
        Compiles the initial mathematical model from a sequence of training sample inputs.
        """
        compiled_vectors = [self.ingest_and_transform_signal(sample) for sample in baseline_audio_samples]
        self.signature_history = compiled_vectors[-self.horizon:]
        
        # Compute baseline threshold variance between initial training matrices
        matrix_variance = np.var(compiled_vectors, axis=0)
        self.adaptive_threshold = float(np.mean(matrix_variance) * 2.5) + 0.02
        print(f"[ENGINE INITIALIZATION] Core Motif Anchored. Adaptive Variance Gate: {self.adaptive_threshold:.6f}")

    def assert_state_activation(self, live_audio_frame):
        """
        [CREDIT / STATE TRANSITION] Asserts a live assertion test pass.
        Calculates the mathematical distance between incoming data and the evolving state memory.
        If verified, the script updates its internal parameters to match natural biometric evolution.
        """
        if not self.signature_history:
            raise ValueError("Engine non-operational: Initialize baseline motif registration matrix prior to execution.")
            
        # 1. Transform raw spatial wave into frequency space
        live_vector = self.ingest_and_transform_signal(live_audio_frame)
        
        # 2. Compute Mean Absolute Error (MAE) distance against historical baseline configurations
        historical_matrix = np.array(self.signature_history)
        mean_historical_profile = np.mean(historical_matrix, axis=0)
        
        structural_distance = float(np.mean(np.abs(live_vector - mean_historical_profile)))
        
        # 3. Dynamic Threshold Assertion Check
        if structural_distance <= self.adaptive_threshold:
            # SUCCESS: Signature match verified.
            # Evolving feedback loop: Append current verified entry into history matrix 
            # and purge oldest data to allow tracking of long-term biometric changes.
            self.signature_history.append(live_vector)
            if len(self.signature_history) > self.horizon:
                self.signature_history.pop(0)
                
            # Recalculate adaptive threshold based on fresh history matrix variance
            updated_variance = np.var(self.signature_history, axis=0)
            self.adaptive_threshold = float(np.mean(updated_variance) * 2.5) + 0.02
            
            return True, structural_distance, "[STATE FINALIZED] Verification matched. State machine executed cleanly."
        else:
            # FAILURE: Signal variation falls outside acceptable tolerance thresholds
            return False, structural_distance, "[REJECTION ALERT] Frequency mismatch or unauthorized signature attempt."

# ==========================================
# SIMULATION WORKFLOW RUNNER
# ==========================================
if __name__ == "__main__":
    print("[*] Launching Self-Evolving Signal Orchestration Engine...")
    
    # 1. Synthesize mock calibration inputs (e.g., child playing a custom three-note motif)
    # Each sample consists of a vectorized frequency wave with subtle random environmental noise
    base_frequency_pattern = np.sin(np.linspace(0, 2 * np.pi * 5, 128))
    
    training_data = [base_frequency_pattern + np.random.normal(0, 0.01, 128) for _ in range(5)]
    
    # Instantiate engine core
    engine = AdaptiveSovereignEngine(frequency_lanes=128, history_horizon=5)
    engine.register_initial_motif(training_data)
    
    # 2. Test execution pass 1: Authentic user input with slight stylistic variation
    authentic_test_signal = base_frequency_pattern + np.random.normal(0, 0.015, 128)
    matched, distance, feedback = engine.assert_state_activation(authentic_test_signal)
    print(f"\n[Run 1: Authentic Signature] -> Verified: {matched} | Distance Score: {distance:.6f}\nLog: {feedback}")
    
    # 3. Test execution pass 2: Unauthorized entry attempt using mismatched frequency profiles
    unauthorized_test_signal = np.cos(np.linspace(0, 2 * np.pi * 12, 128))
    matched, distance, feedback = engine.assert_state_activation(unauthorized_test_signal)
    print(f"\n[Run 2: Counterfeit/Noise Input] -> Verified: {matched} | Distance Score: {distance:.6f}\nLog: {feedback}")
