import mongoose from "mongoose";

const otpSchema = new mongoose.Schema(
{
    otp: {
        type: String,
        default: null,
    },
    otpExpiresAt: {
        type: Date,
        default: null,
        index: { expires: "5m" },
    },
    isVerified: {
        type: String,
        enum: ["verified", "pending"],
        required: true,
        default: "pending",
    },
})

export const Otp = new mongoose.model("Otp" , otpSchema)