import datetime
import os
import random
import string
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, InputFile
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler
import torch
from diffusers import StableDiffusionPipeline

BOT_TOKEN = "telegram-bot-api-token" # replace with your telegram bot api. Make sure the privacy mode is disabled.
IMAGE_PATH = "Images/image.png"
USER_DATA_ROOT = "Omora-Ai"
INITIAL_BALANCE = 0.5  # Initial balance for verified users
GENERATION_COST = 0.3  # Cost per image generation

# Initialize the pipeline once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = None

def initialize_pipeline():
    global pipe
    if pipe is None:
        pipe = StableDiffusionPipeline.from_pretrained(
            "SG161222/Realistic_Vision_V3.0_VAE",
            torch_dtype=torch.float32,
            safety_checker=True,
            feature_extractor=None,
            use_safetensors=True
        )
        pipe.text_encoder.to(device)
        pipe.vae.to(device)
        pipe.unet.to(device)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=False)

def get_greeting():
    h = datetime.datetime.now().hour
    if 5 <= h < 12: return "Good Morning  ☕"
    if 12 <= h < 17: return "Good Afternoon 🍔"
    if 17 <= h < 20: return "Good Evening 🥐"
    if 20 <= h < 24: return "Good Night 🌜"
    return "Midnight ☄️"

def check_user_verification(user_id):
    if not os.path.exists(f"{USER_DATA_ROOT}/userdata.txt"):
        return False, None, 0.0
    
    with open(f"{USER_DATA_ROOT}/userdata.txt", 'r') as file:
        for line in file:
            if f"{user_id} - Age - Verified" in line:
                # Extract balance from the line if it exists
                if "Money =" in line:
                    balance_str = line.split("Money =")[1].split("$")[0].strip()
                    try:
                        balance = float(balance_str)
                    except ValueError:
                        balance = INITIAL_BALANCE
                else:
                    balance = INITIAL_BALANCE
                return True, "Verified", balance
            elif f"{user_id} - Age - Not Verified" in line:
                return True, "Not Verified", 0.0
    return False, None, 0.0

def get_user_setting(user_id, setting_name, default_value=None):
    user_folder = f"{USER_DATA_ROOT}/{user_id}"
    user_file = f"{user_folder}/settings.txt"
    
    if not os.path.exists(user_file):
        return default_value
    
    with open(user_file, 'r') as file:
        for line in file:
            if f"{setting_name} - " in line:
                return line.split(f"{setting_name} - ")[1].strip()
    return default_value

def save_user_data(user_id, status):
    os.makedirs(USER_DATA_ROOT, exist_ok=True)
    
    # Check if user already exists to preserve their balance
    current_balance = 0.0
    if status == "Verified":
        _, _, existing_balance = check_user_verification(user_id)
        if existing_balance > 0:
            current_balance = existing_balance
        else:
            current_balance = INITIAL_BALANCE
    
    if status == "Verified":
        entry = f"{{{user_id} - Age - Verified, Money = {current_balance}$}};\n"
    else:
        entry = f"{{{user_id} - Age - Not Verified}};\n"
    
    if os.path.exists(f"{USER_DATA_ROOT}/userdata.txt"):
        with open(f"{USER_DATA_ROOT}/userdata.txt", 'r') as file:
            lines = file.readlines()
        
        user_exists = False
        for i, line in enumerate(lines):
            if str(user_id) in line and "Age" in line:
                lines[i] = entry
                user_exists = True
                break
        
        if not user_exists:
            lines.append(entry)
        
        with open(f"{USER_DATA_ROOT}/userdata.txt", 'w') as file:
            file.writelines(lines)
    else:
        with open(f"{USER_DATA_ROOT}/userdata.txt", 'w') as file:
            file.write(entry)

def update_user_balance(user_id, amount):
    is_registered, status, current_balance = check_user_verification(user_id)
    if not is_registered or status != "Verified":
        return False
    
    new_balance = current_balance + amount
    
    if os.path.exists(f"{USER_DATA_ROOT}/userdata.txt"):
        with open(f"{USER_DATA_ROOT}/userdata.txt", 'r') as file:
            lines = file.readlines()
        
        for i, line in enumerate(lines):
            if f"{user_id} - Age - Verified" in line:
                lines[i] = f"{{{user_id} - Age - Verified, Money = {new_balance}$}};\n"
                break
        
        with open(f"{USER_DATA_ROOT}/userdata.txt", 'w') as file:
            file.writelines(lines)
        
        return True
    return False

def save_user_setting(user_id, setting_name, value):
    user_folder = f"{USER_DATA_ROOT}/{user_id}"
    os.makedirs(user_folder, exist_ok=True)
    user_file = f"{user_folder}/settings.txt"
    
    entry = f"{setting_name} - {value}\n"
    
    if os.path.exists(user_file):
        with open(user_file, 'r') as file:
            lines = file.readlines()
        
        setting_exists = False
        for i, line in enumerate(lines):
            if setting_name in line:
                lines[i] = entry
                setting_exists = True
                break
        
        if not setting_exists:
            lines.append(entry)
        
        with open(user_file, 'w') as file:
            file.writelines(lines)
    else:
        with open(user_file, 'w') as file:
            file.write(entry)

def generate_random_filename(user_id):
    user_images_folder = f"{USER_DATA_ROOT}/{user_id}/Images"
    os.makedirs(user_images_folder, exist_ok=True)
    
    while True:
        random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        filename = f"image_{random_str}.png"
        if not os.path.exists(f"{user_images_folder}/{filename}"):
            return filename

def save_prompt_to_history(user_id, image_filename, prompt):
    user_images_folder = f"{USER_DATA_ROOT}/{user_id}/Images"
    os.makedirs(user_images_folder, exist_ok=True)
    
    history_file = f"{user_images_folder}/prompt.txt"
    
    with open(history_file, 'a') as file:
        file.write(f"{image_filename} - {prompt}\n")

async def imagine_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    
    # Check verification and balance
    is_registered, status, balance = check_user_verification(user.id)
    if not is_registered:
        await age_verification(update, context)
        return
    elif status != "Verified":
        await update.message.reply_text("🚫 Sorry, you must be 18+ to use this bot.")
        return
    
    if balance < GENERATION_COST:
        await update.message.reply_text(f"❌ Insufficient balance. You need ${GENERATION_COST} to generate an image. Your current balance: ${balance:.1f}")
        return
    
    # Get the prompt from the message
    if not context.args:
        await update.message.reply_text("ℹ️ Please provide a prompt after the /imagine command.")
        return
    
    prompt = ' '.join(context.args)
    
    # Get user settings or use defaults
    inference_steps = int(get_user_setting(user.id, "InferenceSteps", "20"))
    guidance_scale = float(get_user_setting(user.id, "GuideScale", "7.5"))
    image_size = get_user_setting(user.id, "ImageSize", "512x512")
    
    # Parse image size
    try:
        width, height = map(int, image_size.split('x'))
    except:
        width, height = 512, 512
    
    # Deduct balance
    if not update_user_balance(user.id, -GENERATION_COST):
        await update.message.reply_text("❌ Error updating your balance. Please try again.")
        return
    
    # Send processing message
    processing_msg = await update.message.reply_text("🔄 Generating your image... This may take a moment.")
    
    try:
        # Generate the image
        initialize_pipeline()
        image = pipe(
            prompt=prompt,
            negative_prompt="blurry, cartoon, deformed, bad anatomy, low quality",
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height
        ).images[0]
        
        # Save the image
        filename = generate_random_filename(user.id)
        user_images_folder = f"{USER_DATA_ROOT}/{user.id}/Images"
        os.makedirs(user_images_folder, exist_ok=True)
        image_path = f"{user_images_folder}/{filename}"
        image.save(image_path)
        
        # Save prompt to history
        save_prompt_to_history(user.id, filename, prompt)
        
        # Send the image as a file without compression
        with open(image_path, "rb") as photo:
            await context.bot.send_document(
                chat_id=update.effective_chat.id,
                document=InputFile(photo),
                filename=filename,
                reply_to_message_id=update.message.message_id
            )
        
        # Delete processing message
        await processing_msg.delete()
        
    except Exception as e:
        print(f"❌ Error in imagine_command: {e}")
        # Refund the user if generation failed
        update_user_balance(user.id, GENERATION_COST)
        await update.message.reply_text("❌ An error occurred while generating your image. Your balance has been refunded.")

async def age_verification(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    
    is_registered, status, balance = check_user_verification(user.id)
    if is_registered:
        if status == "Verified":
            await start(update, context)
        else:
            await update.message.reply_text("🚫 Sorry, you must be 18+ to use this bot.")
        return
    
    keyboard = [
        [InlineKeyboardButton("18+", callback_data="age_verified"),
         InlineKeyboardButton("13+", callback_data="age_not_verified")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "🔞 Age Verification\n\nPlease select your age group to continue:",
        reply_markup=reply_markup
    )

async def handle_age_verification(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    user = query.from_user
    choice = query.data
    
    if choice == "age_verified":
        save_user_data(user.id, "Verified")
        await query.message.delete()
        await start(update, context)  # Silently proceed to main menu
    elif choice == "age_not_verified":
        save_user_data(user.id, "Not Verified")
        await query.message.edit_text(
            "🚫 Sorry, you must be 18+ to use this bot.\n\n"
            "Your age has been recorded as 13-17."
        )

async def images_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    is_registered, status, balance = check_user_verification(query.from_user.id)
    if not is_registered or status != "Verified":
        await query.message.reply_text("Please complete age verification first.")
        return
    
    message = (
        "🎨 *Image Generation Guide*\n\n"
        "To create amazing images, use the following command:\n"
        "`/imagine <your prompt>`\n\n"
        "*Example:*\n"
        "`/imagine A cat holding a sign that says 'Omora AI'`\n\n"
        "✨ *Tips:*\n"
        "- Be descriptive with your prompts\n"
        "- Add style preferences (digital art, photorealistic, etc.)\n"
        "- Keep prompts under 400 characters"
    )
    
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="back_to_start")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    try:
        await query.message.delete()
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=message,
            parse_mode="Markdown",
            reply_markup=reply_markup
        )
    except Exception as e:
        print(f"❌ Error in images_callback: {e}")

async def account_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    is_registered, status, balance = check_user_verification(query.from_user.id)
    if not is_registered or status != "Verified":
        await query.message.reply_text("Please complete age verification first.")
        return
    
    user = query.from_user
    message = (
        "📊 <b>User Information</b>\n\n"
        f"🆔 <b>User ID:</b> <code>{user.id}</code>\n"
        f"💰 <b>Balance:</b> ${balance:.1f}\n"
        f"💎 <b>Premium:</b> False\n"
        f"⏳ <b>Premium Expiration Time:</b> NA"
    )
    
    keyboard = [
        [InlineKeyboardButton("💵 Deposit", url="https://example.com"),
         InlineKeyboardButton("💎 Premium", url="https://example.com")],
        [InlineKeyboardButton("🔙 Back", callback_data="back_to_start")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    try:
        await query.message.delete()
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=message,
            parse_mode="HTML",
            reply_markup=reply_markup
        )
    except Exception as e:
        print(f"❌ Error in account_callback: {e}")

async def more_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    is_registered, status, balance = check_user_verification(query.from_user.id)
    if not is_registered or status != "Verified":
        await query.message.reply_text("Please complete age verification first.")
        return
    
    keyboard = [
        [
            InlineKeyboardButton("Inference", callback_data="inference_settings"),
            InlineKeyboardButton("Guide Scale", callback_data="guide_scale_settings")
        ],
        [
            InlineKeyboardButton("Size", callback_data="size_settings"),
            InlineKeyboardButton("🔙 Back", callback_data="back_to_start")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    try:
        await query.message.delete()
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="⚙️ <b>Advanced Settings</b>\n\nSelect an option to configure:",
            parse_mode="HTML",
            reply_markup=reply_markup
        )
    except Exception as e:
        print(f"❌ Error in more_callback: {e}")

async def inference_settings_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    current_setting = get_user_setting(user_id, "InferenceSteps", "20")
    
    keyboard = [
        [
            InlineKeyboardButton("Default (20)", callback_data="inference_20"),
            InlineKeyboardButton("30", callback_data="inference_30")
        ],
        [
            InlineKeyboardButton("40", callback_data="inference_40"),
            InlineKeyboardButton("50", callback_data="inference_50")
        ],
        [InlineKeyboardButton("🔙 Back", callback_data="more_settings")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    try:
        await query.message.delete()
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=f"🛠️ <b>Inference Steps</b>\n\nCurrent setting: {current_setting}\n\nSelect your preferred inference steps:",
            parse_mode="HTML",
            reply_markup=reply_markup
        )
    except Exception as e:
        print(f"❌ Error in inference_settings_callback: {e}")

async def guide_scale_settings_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    current_setting = get_user_setting(user_id, "GuideScale", "7.5")
    
    keyboard = [
        [
            InlineKeyboardButton("Default (7.5)", callback_data="guide_scale_7.5"),
            InlineKeyboardButton("6.5", callback_data="guide_scale_6.5")
        ],
        [
            InlineKeyboardButton("7.5", callback_data="guide_scale_7.5"),
            InlineKeyboardButton("8.5", callback_data="guide_scale_8.5")
        ],
        [InlineKeyboardButton("🔙 Back", callback_data="more_settings")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    try:
        await query.message.delete()
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=f"📏 <b>Guidance Scale</b>\n\nCurrent setting: {current_setting}\n\nSelect your preferred guidance scale:",
            parse_mode="HTML",
            reply_markup=reply_markup
        )
    except Exception as e:
        print(f"❌ Error in guide_scale_settings_callback: {e}")

async def size_settings_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    current_setting = get_user_setting(user_id, "ImageSize", "512x512")
    
    keyboard = [
        [
            InlineKeyboardButton("Default (512x512)", callback_data="size_512x512"),
            InlineKeyboardButton("512x768", callback_data="size_512x768")
        ],
        [
            InlineKeyboardButton("768x512", callback_data="size_768x512"),
            InlineKeyboardButton("768x768", callback_data="size_768x768")
        ],
        [InlineKeyboardButton("🔙 Back", callback_data="more_settings")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    try:
        await query.message.delete()
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=f"🖼️ <b>Image Size</b>\n\nCurrent setting: {current_setting}\n\nSelect your preferred image size:",
            parse_mode="HTML",
            reply_markup=reply_markup
        )
    except Exception as e:
        print(f"❌ Error in size_settings_callback: {e}")

async def handle_setting_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    choice = query.data
    
    if choice.startswith("inference_"):
        steps = choice.split("_")[1]
        save_user_setting(user_id, "InferenceSteps", steps)
        await query.edit_message_text(
            text=f"✅ Inference steps set to: {steps}",
            parse_mode="HTML"
        )
        await inference_settings_callback(update, context)
    elif choice.startswith("guide_scale_"):
        scale = choice.split("_")[2]
        save_user_setting(user_id, "GuideScale", scale)
        await query.edit_message_text(
            text=f"✅ Guidance scale set to: {scale}",
            parse_mode="HTML"
        )
        await guide_scale_settings_callback(update, context)
    elif choice.startswith("size_"):
        size = choice.split("_")[1]
        save_user_setting(user_id, "ImageSize", size)
        await query.edit_message_text(
            text=f"✅ Image size set to: {size}",
            parse_mode="HTML"
        )
        await size_settings_callback(update, context)

async def back_to_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    user = query.from_user
    greeting = get_greeting()

    keyboard = [
        [InlineKeyboardButton("Account 👤", callback_data="account_info"),
         InlineKeyboardButton("Support 🧏‍♂️", url="https://t.me/your_support_link")],
        [InlineKeyboardButton("➕ Generate with Me in Your Group", url="https://t.me/your_bot_username?startgroup=true")],
        [InlineKeyboardButton("Images 🖼️", callback_data="images_guide"),
         InlineKeyboardButton("More ⚙️", callback_data="more_settings")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    zw = "\u200B"
    caption = (
        f"👋 Hey {user.first_name}, {greeting}\n"
        "🧠 I'm an AI bot that turns your ideas into images.\n\n"
        f"{zw}<blockquote>Just add me a description of what you imagine... and I'll bring it to life!</blockquote>"
    )

    try:
        await query.message.delete()
        with open(IMAGE_PATH, "rb") as photo:
            await context.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=InputFile(photo),
                caption=caption,
                parse_mode="HTML",
                reply_markup=reply_markup
            )
    except Exception as e:
        print(f"❌ Error in back_to_start: {e}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    
    is_registered, status, balance = check_user_verification(user.id)
    
    if not is_registered:
        await age_verification(update, context)
        return
    elif status != "Verified":
        await update.message.reply_text("🚫 Sorry, you must be 18+ to use this bot.")
        return
    
    greeting = get_greeting()

    keyboard = [
        [InlineKeyboardButton("Account 👤", callback_data="account_info"),
         InlineKeyboardButton("Support 🧏‍♂️", url="https://t.me/your_support_link")],
        [InlineKeyboardButton("➕ Generate with Me in Your Group", url="https://t.me/your_bot_username?startgroup=true")],
        [InlineKeyboardButton("Images 🖼️", callback_data="images_guide"),
         InlineKeyboardButton("More ⚙️", callback_data="more_settings")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    zw = "\u200B"
    caption = (
        f"👋 Hey {user.first_name}, {greeting}\n"
        "🧠 I'm an AI bot that turns your ideas into images.\n\n"
        f"{zw}<blockquote>Just add me a description of what you imagine... and I'll bring it to life!</blockquote>"
    )

    try:
        with open(IMAGE_PATH, "rb") as photo:
            await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=InputFile(photo),
                caption=caption,
                parse_mode="HTML",
                reply_markup=reply_markup
            )
    except Exception as e:
        print(f"❌ Error in start: {e}")

if __name__ == "__main__":
    os.makedirs(USER_DATA_ROOT, exist_ok=True)
    
    app = Application.builder().token(BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("imagine", imagine_command))
    app.add_handler(CallbackQueryHandler(handle_age_verification, pattern="^age_(verified|not_verified)$"))
    app.add_handler(CallbackQueryHandler(account_callback, pattern="^account_info$"))
    app.add_handler(CallbackQueryHandler(images_callback, pattern="^images_guide$"))
    app.add_handler(CallbackQueryHandler(back_to_start, pattern="^back_to_start$"))
    app.add_handler(CallbackQueryHandler(more_callback, pattern="^more_settings$"))
    app.add_handler(CallbackQueryHandler(inference_settings_callback, pattern="^inference_settings$"))
    app.add_handler(CallbackQueryHandler(guide_scale_settings_callback, pattern="^guide_scale_settings$"))
    app.add_handler(CallbackQueryHandler(size_settings_callback, pattern="^size_settings$"))
    app.add_handler(CallbackQueryHandler(handle_setting_selection, pattern="^(inference|guide_scale|size)_"))
    
    print("🤖 Bot is running...")
    print(f"📝 User data will be saved in: {USER_DATA_ROOT}")
    app.run_polling()