class MenuHandler:
    @staticmethod
    def display_main_menu():
        print("\n=== Attendance System Menu ===")
        print("1. Start New Session")
        print("2. Verify Attendance")
        print("3. End Session")
        print("4. Test Verification")
        print("5. Register New Student")
        print("6. Exit")

    @staticmethod
    def display_registration_instructions():
        print("\n=== Student Registration Instructions ===")
        print("1. Fill in the required student information")
        print("2. When ready to capture face image:")
        print("   - Ensure good lighting")
        print("   - Look directly at the camera")
        print("   - Keep a neutral expression")
        print("3. Press SPACE to capture or Q to quit")

    @staticmethod
    def display_verification_instructions():
        print("\n=== Face Verification Instructions ===")
        print("1. Position your face within the green box")
        print("2. Ensure good lighting conditions")
        print("3. Look directly at the camera")
        print("4. Keep a neutral expression")
        print("5. Stay still during capture")
        print("6. Press SPACE to capture or Q to quit")

    @staticmethod
    def get_user_input(prompt: str) -> str:
        return input(prompt).strip()