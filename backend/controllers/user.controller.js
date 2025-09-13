import crypto from "crypto";
import jwt from "jsonwebtoken";
import { ApiError } from "../utils/ApiError.utils.js";
import { ApiResponse } from "../utils/ApiResponse.utils.js";
import { asyncHandler } from "../utils/asyncHandler.utils.js";
import { sendEmail } from "../utils/sendEmail.utils.js";
import { User } from "../models/user.model.js";
import { Otp } from "../models/otp.model.js";

const option = {
  httpOnly: true,
  secure: true,
};

const generateAccessAndRefreshToken = async (userId) => {
  try {
    const user = await User.findById(userId);
    const accessToken = user.generateAccessToken();
    const refreshToken = user.generateRefreshToken();

    user.refreshToken = refreshToken;
    await user.save({ validateBeforeSave: false });

    return { accessToken, refreshToken };
  } catch (error) {
    throw new ApiError(
      500,
      "Something went wrong while generating access and refresh token"
    );
  }
};

const register = asyncHandler(async (req, res) => {
  const { username, email, phone, password, googleId } = req.body;

  let user = await User.findOne({ $or: [{ email }, { phone }] });

  if (user) throw new ApiError(400, "User already exists");

  user = await User.create({
    username: username.toLowerCase(),
    email,
    phone,
    password,
    googleId,
  });

  const { accessToken, refreshToken } = await generateAccessAndRefreshToken(
    user._id
  );

  const createdUser = await User.findById(user._id).select(
    "-password -refreshToken"
  );

  if (!createdUser) {
    throw new ApiError("Something went wrong while registering the user");
  }

  return res
    .status(200)
    .cookie("accessToken", accessToken, option)
    .cookie("refreshToken", refreshToken, option)
    .json(
      new ApiResponse(
        200,
        { user: createdUser, accessToken, refreshToken },
        "User Registered Successfully"
      )
    );
});

const login = asyncHandler(async (req, res) => {
  const { username, email, password } = req.body;

  if (!(username || email))
    throw new ApiError(400, "Username or Email must be required");

  const user = await User.findOne({ $or: [{ email }, { username }] });

  if (!user) throw new ApiError(404, "User does not exists");

  const isPasswordCorrect = await user.comparePassword(password);

  if (!isPasswordCorrect) throw new ApiError(401, "Invalid User Credentials");

  const { accessToken, refreshToken } = await generateAccessAndRefreshToken(
    user._id
  );

  const loggedInUser = await User.findById(user._id).select(
    "-password -refreshToken"
  );

  if (!loggedInUser)
    throw new ApiError(404, "Something went wrong while log in");

  return res
    .status(200)
    .cookie("accessToken", accessToken, option)
    .cookie("refreshToken", refreshToken, option)
    .json(
      new ApiResponse(
        200,
        { user: loggedInUser, accessToken, refreshToken },
        "User Login Successfully"
      )
    );
});

const logout = asyncHandler(async (req, res) => {
  await User.findByIdAndUpdate(req.user._id, {
    $set: {
      refreshToken: undefined,
    },
    new: true,
  });

  const option = {
    httpOnly: true,
    secure: true,
  };

  return res
    .status(200)
    .clearCookie("accessToken", option)
    .clearCookie("refreshToken", option)
    .json(new ApiResponse(200, "User Logged Out"));
});

const sendOtpToEmail = asyncHandler(async (req, res) => {
  const { username, email, phone, password } = req.user || req.body;

  if (!email) throw new ApiError(400, "Email is required");

  if (req.googleId) {
    await sendEmail(
      email,
      "Login Successful",
      `Login Successful in Ticket-Sync`
    );
  } else {
    const tempotp = crypto.randomInt(100000, 999999).toString();

    const otpExpiresAt = new Date(Date.now() + 5 * 60 * 1000);

    await sendEmail(
      email,
      "Your OTP Code",
      `Your OTP is for Ticket-Sync: ${tempotp}`
    );

    const votp = await Otp.create({
      otp : tempotp,
      otpExpiresAt,
    })
  }

  return res.status(200).json(new ApiResponse(200, "Email sent successfully"));
});

const verifyOtp = asyncHandler(async (req, res) => {
  const { otp } = req.body;

  // const user = await User.findOne({ email });
  const votp = await Otp.findOne({otp});

  // if (!user.googleId) {
    if (!votp || votp.otpExpiresAt < new Date()) {
      throw new ApiError(400, "Invalid or expired OTP");
    }

    votp.otp = null;
    votp.otpExpiresAt = null;
  // }
    votp.isVerified = "verified";
    await votp.save();

  return res.status(200).json(new ApiResponse(200, "verified successfully"));
});

const refreshAccessToken = asyncHandler(async (req, res) => {
  try {
    const incomingRefreshToken =
      req.cookies.refreshToken || req.body.refreshToken;

    if (!incomingRefreshToken) {
      throw new ApiError(401, "Unauthorized request");
    }

    const decodedToken = jwt.verify(
      incomingRefreshToken,
      process.env.REFRESH_TOKEN_SECRET
    );

    const user = await User.findById(decodedToken?._id);

    if (!user) {
      throw new ApiError(401, "Invalid refresh token");
    }

    if (!(incomingRefreshToken === user?.refreshToken)) {
      throw new ApiError(401, "Refresh token is expired or Used");
    }

    const { accessToken: newAccessToken, refreshToken: newRefreshToken } =
      await generateAccessAndRefreshToken(user._id);

    const option = {
      httpOnly: true,
      secure: true,
    };

    return res
      .status(200)
      .cookie("accessToken", newAccessToken, option)
      .cookie("refreshToken", newRefreshToken, option)
      .json(
        new ApiResponse(
          200,
          { accessToken: newAccessToken, refreshToken: newRefreshToken },
          "Access Token Refreshed"
        )
      );
  } catch (error) {
    throw new ApiError(401, error?.message || "Invalid Refresh Token");
  }
});

const googleAuth = async (req, res) => {
  const { accessToken, refreshToken } = req.user;

  const userId = req.user._id;

  console.log("Authenticated User:", req.user);

  const user = await User.findById(userId).select("-password -refreshToken");

  if (!user) throw new ApiError(404, "User does not exists");

  return res
    .status(200)
    .cookie("accessToken", accessToken, option)
    .cookie("refreshToken", refreshToken, option)
    .json(
      new ApiResponse(
        200,
        { user, accessToken, refreshToken },
        "Google Login successful"
      )
    );
};

export {
  register,
  login,
  logout,
  sendOtpToEmail,
  verifyOtp,
  refreshAccessToken,
  googleAuth,
};
